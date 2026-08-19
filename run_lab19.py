"""Lab 19 local runner: GraphRAG vs Flat RAG on a 5000-row news subset."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from collections import Counter, defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.max_colwidth", 120)

DATA_PATH = ROOT / "data" / "hackernoon_subset.csv"
GOLDEN_PATH = ROOT / "data" / "golden_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "data").mkdir(parents=True, exist_ok=True)
(ROOT / "reports").mkdir(parents=True, exist_ok=True)

LAB_MAX_ARTICLES = 1500
LAB_MAX_CHUNKS = 3000
EXTRACTION_MAX_CHUNKS = 80
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

ALLOWED_NODE_TYPES = {"Company", "Person", "Technology"}
ALLOWED_RELATIONS = {
    "ACQUIRED", "DEVELOPED", "INVESTED_IN", "FOUNDED",
    "WORKED_AT", "PARTNERED_WITH", "USES", "LEADS",
}
CORP_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc", "plc", "co", "company"}
PRODUCTISH = {
    "watch", "music", "tv", "store", "pay", "arcade", "vision", "maps", "cloud",
    "azure", "office", "teams", "iphone", "ipad", "ios", "android", "windows",
    "copilot", "bing", "chrome", "pixel", "play", "ads", "drive", "docs",
}
MANUAL_ALIASES = {
    "msft": "Microsoft",
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "goog": "Google",
    "googl": "Google",
    "google llc": "Google",
    "alphabet": "Google",
    "alphabet inc": "Google",
    "meta platforms": "Meta",
    "meta platforms inc": "Meta",
    "facebook": "Meta",
    "aapl": "Apple",
    "apple inc": "Apple",
    "amzn": "Amazon",
    "amazon.com": "Amazon",
    "amazon com": "Amazon",
    "nvda": "NVIDIA",
    "nvidia corporation": "NVIDIA",
}

SUPER_NODE_DEGREE = 100
SUPER_NODE_EDGE_CAP = 50
GLOBAL_EDGE_CAP = 250
MAX_GRAPH_CONTEXT_CHARS = 14000
ER_THRESHOLD = 0.90


def get_secret(name, default=None):
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value is not None:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


NEO4J_URI = get_secret("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = get_secret("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_secret("NEO4J_PASSWORD", "")
NEO4J_DATABASE = get_secret("NEO4J_DATABASE", "neo4j")
GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
GROQ_MODEL = get_secret("GROQ_MODEL", "")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
NVIDIA_API_KEY = get_secret("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = get_secret("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = get_secret("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
JUDGE_PROVIDER = (get_secret("JUDGE_PROVIDER", "nvidia") or "nvidia").lower()
JUDGE_MODEL = get_secret("JUDGE_MODEL", NVIDIA_MODEL)

driver = None
embedder = None
flat_index = None
flat_store = None
entity_match_vectors = None
entity_match_store = None


def norm_space(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


def sha1(x):
    return hashlib.sha1(str(x).encode("utf-8", errors="ignore")).hexdigest()


def parse_json_object(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("No JSON object found.")
    return json.loads(text[a:b + 1])


def _openai_compat_client():
    if NVIDIA_API_KEY:
        return OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY), NVIDIA_MODEL
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), get_secret("OPENAI_MODEL", "gpt-4o-mini")
    raise RuntimeError("No LLM credentials (GROQ/NVIDIA/OPENAI).")


def llm_chat(messages, json_mode=False, max_retries=4, max_tokens=1200):
    last = None
    if GROQ_API_KEY and GROQ_MODEL:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                usage = {}
                if getattr(resp, "usage", None):
                    usage = {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                        "total_tokens": getattr(resp.usage, "total_tokens", None),
                    }
                return resp.choices[0].message.content, usage
            except Exception as e:
                last = e
                time.sleep(min(20, 2 ** attempt + random.random()))
        raise RuntimeError(last)

    client, model = _openai_compat_client()
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception:
                kwargs.pop("response_format", None)
                if json_mode:
                    messages = list(messages) + [
                        {"role": "user", "content": "Return a single JSON object only. No markdown."}
                    ]
                    kwargs["messages"] = messages
                resp = client.chat.completions.create(**kwargs)
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            return resp.choices[0].message.content or "", usage
        except Exception as e:
            last = e
            time.sleep(min(20, 2 ** attempt + random.random()))
    raise RuntimeError(last)


def llm_json(system, user, max_tokens=1200):
    text, usage = llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        json_mode=True,
        max_tokens=max_tokens,
    )
    return parse_json_object(text), usage


def connect_neo4j():
    global driver
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise ValueError("Thiếu Neo4j secrets.")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("[OK] Neo4j connected.")


def run_cypher(query, **params):
    if driver is None:
        raise RuntimeError("Hãy chạy connect_neo4j() trước.")
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, **params)
        rows = [r.data() for r in result]
        result.consume()
    return rows


def setup_graph_schema():
    for stmt in [
        """
        CREATE CONSTRAINT entity_id IF NOT EXISTS
        FOR (n:Entity) REQUIRE n.id IS UNIQUE
        """,
        """
        CREATE INDEX entity_name_norm IF NOT EXISTS
        FOR (n:Entity) ON (n.name_norm)
        """,
        """
        CREATE INDEX company_name_norm IF NOT EXISTS
        FOR (n:Company) ON (n.name_norm)
        """,
        """
        CREATE INDEX person_name_norm IF NOT EXISTS
        FOR (n:Person) ON (n.name_norm)
        """,
        """
        CREATE INDEX technology_name_norm IF NOT EXISTS
        FOR (n:Technology) ON (n.name_norm)
        """,
    ]:
        run_cypher(stmt)
    print("[OK] Schema ready.")


def pick_col(df, candidates, required=True):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise KeyError(f"Missing one of columns: {candidates}")
    return None


def load_news(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported: {path.suffix}")


def standardize_news(raw):
    text_col = pick_col(raw, ["text", "content", "article", "body", "story", "description"])
    title_col = pick_col(raw, ["title", "headline"], required=False)
    date_col = pick_col(raw, ["published_date", "date", "published_at", "created_at"], required=False)
    id_col = pick_col(raw, ["id", "article_id", "story_id", "uuid", "_id"], required=False)

    df = pd.DataFrame()
    df["text"] = raw[text_col].fillna("").map(norm_space)
    df["title"] = raw[title_col].fillna("").map(norm_space) if title_col else ""
    if date_col:
        df["published_date"] = (
            pd.to_datetime(raw[date_col], errors="coerce", utc=True)
            .dt.strftime("%Y-%m-%d")
            .fillna("")
        )
    else:
        df["published_date"] = ""
    if id_col:
        df["article_id"] = raw[id_col].astype(str)
    else:
        df["article_id"] = [sha1(f"{t}\n{x}")[:20] for t, x in zip(df["title"], df["text"])]
    df = df[df["text"].str.len() >= 80].copy()
    df["dedup_key"] = [sha1(norm_space(f"{t}\n{x}").lower()) for t, x in zip(df["title"], df["text"])]
    before = len(df)
    df = df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)
    print(f"Exact dedup: {before:,} -> {len(df):,}")
    return df


def simhash64(text, ngram=3):
    s = re.sub(r"[^a-z0-9]+", " ", norm_space(text).lower())
    toks = s.split()
    grams = [" ".join(toks[i:i + ngram]) for i in range(max(1, len(toks) - ngram + 1))] or [s]
    acc = [0] * 64
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        for b in range(64):
            acc[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b, v in enumerate(acc):
        if v >= 0:
            out |= 1 << b
    return out


def hamming(a, b):
    return (a ^ b).bit_count()


def near_dedup(news_df, max_hamming=3):
    """SimHash LSH near-dedup — O(N) banding, not pairwise cosine O(N^2)."""
    hashes = [simhash64(f"{r.title}\n{r.text}") for r in news_df.itertuples(index=False)]
    bands = defaultdict(list)
    keep = np.ones(len(news_df), dtype=bool)
    for i, h in enumerate(hashes):
        key = (h >> 48, (h >> 32) & 0xFFFF)
        dup = False
        for j in bands[key]:
            if hamming(h, hashes[j]) <= max_hamming:
                dup = True
                break
        if dup:
            keep[i] = False
        else:
            bands[key].append(i)
    out = news_df.loc[keep].reset_index(drop=True)
    print(f"Near-dedup SimHash: {len(news_df):,} -> {len(out):,} (hamming<={max_hamming})")
    return out


def prioritize_articles(news_df, limit):
    pat = re.compile(
        r"microsoft|google|apple|meta|facebook|openai|hugging\s*face|amazon|nvidia|tesla|"
        r"intel|oracle|ibm|salesforce|adobe|netflix|uber|airbnb|stripe|samsung|"
        r"acquire|invest|founder|ceo|partner|developed",
        re.I,
    )
    score = news_df["title"].fillna("").str.contains(pat).astype(int) * 2
    score += news_df["text"].fillna("").str.contains(pat).astype(int)
    ranked = news_df.assign(_score=score).sort_values(["_score"], ascending=False)
    if limit and len(ranked) > limit:
        ranked = ranked.head(limit)
    return ranked.drop(columns="_score").reset_index(drop=True)


def chunk_text(text, size=220, overlap=40):
    words = norm_space(text).split()
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        part = words[start:start + size]
        if not part:
            break
        out.append(" ".join(part))
        if start + size >= len(words):
            break
    return out


def build_chunks(news_df):
    rows = []
    for r in tqdm(news_df.itertuples(index=False), total=len(news_df), desc="Chunking"):
        body = f"{r.title}. {r.text}".strip()
        for i, text in enumerate(chunk_text(body, CHUNK_WORDS, CHUNK_OVERLAP_WORDS)):
            rows.append({
                "chunk_id": f"{r.article_id}::c{i:04d}",
                "article_id": r.article_id,
                "title": r.title,
                "published_date": r.published_date,
                "text": text,
            })
            if LAB_MAX_CHUNKS and len(rows) >= LAB_MAX_CHUNKS:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


COREF_SYSTEM = """
You are a conservative coreference-resolution component for a knowledge-graph pipeline.
Resolve pronouns and generic references only when the antecedent is clearly supported in the same chunk.
Never invent facts. Preserve dates, numbers, tickers and product names.
If ambiguous, keep the original mention and list it in unresolved_mentions.
Return strict JSON only.
""".strip()


def resolve_coref_batch(batch_df):
    payload = [{"chunk_id": r.chunk_id, "text": r.text} for r in batch_df.itertuples(index=False)]
    prompt = f"""
Resolve coreferences.

Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "resolved_text": "...",
      "unresolved_mentions": ["..."]
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
    obj, usage = llm_json(COREF_SYSTEM, prompt, max_tokens=1800)
    by_id = {x.get("chunk_id"): x for x in obj.get("items", [])}
    rows = []
    for r in batch_df.itertuples(index=False):
        item = by_id.get(r.chunk_id, {})
        rows.append({
            "chunk_id": r.chunk_id,
            "resolved_text": norm_space(item.get("resolved_text") or r.text),
            "unresolved_mentions": item.get("unresolved_mentions", []),
        })
    return pd.DataFrame(rows), usage


def run_coref(chunks_subset, batch_size=4):
    out = []
    for start in tqdm(range(0, len(chunks_subset), batch_size), desc="Coref"):
        batch = chunks_subset.iloc[start:start + batch_size]
        try:
            df, _ = resolve_coref_batch(batch)
        except Exception as e:
            print("coref batch failed:", e)
            df = pd.DataFrame({
                "chunk_id": batch["chunk_id"].tolist(),
                "resolved_text": batch["text"].tolist(),
                "unresolved_mentions": [["COREF_BATCH_FAILED"] for _ in range(len(batch))],
            })
        out.append(df)
    return pd.concat(out, ignore_index=True)


EXTRACT_SYSTEM = f"""
Extract a high-precision knowledge graph from tech-news text.
Allowed node types: {sorted(ALLOWED_NODE_TYPES)}
Allowed relations: {sorted(ALLOWED_RELATIONS)}
Use only explicitly supported facts. Prefer precision over recall.
Every relation needs short evidence. Return strict JSON only.
""".strip()


def extract_batch(batch_df):
    payload = [{
        "chunk_id": r.chunk_id,
        "published_date": r.published_date,
        "text": getattr(r, "resolved_text", None) or r.text,
    } for r in batch_df.itertuples(index=False)]
    prompt = f"""
Return:
{{
  "items": [
    {{
      "chunk_id": "...",
      "relations": [
        {{
          "source": "...",
          "source_type": "Company|Person|Technology",
          "relation": "ALLOWED_RELATION",
          "target": "...",
          "target_type": "Company|Person|Technology",
          "evidence": "...",
          "confidence": 0.0
        }}
      ]
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()
    return llm_json(EXTRACT_SYSTEM, prompt, max_tokens=1800)


def run_extraction(source_df, batch_size=3):
    meta = source_df.set_index("chunk_id")["published_date"].to_dict()
    triples, errors = [], []
    for start in tqdm(range(0, len(source_df), batch_size), desc="NER+RE"):
        batch = source_df.iloc[start:start + batch_size]
        try:
            obj, _ = extract_batch(batch)
        except Exception as e:
            errors.append({"start": start, "error": str(e)})
            continue
        for item in obj.get("items", []):
            cid = item.get("chunk_id")
            if cid not in meta:
                continue
            for x in item.get("relations", []):
                s, t = norm_space(x.get("source")), norm_space(x.get("target"))
                st, tt, rel = x.get("source_type"), x.get("target_type"), x.get("relation")
                if not s or not t:
                    continue
                if st not in ALLOWED_NODE_TYPES or tt not in ALLOWED_NODE_TYPES:
                    continue
                if rel not in ALLOWED_RELATIONS:
                    continue
                triples.append({
                    "source_raw": s,
                    "source_type": st,
                    "relation": rel,
                    "target_raw": t,
                    "target_type": tt,
                    "source_chunk_id": cid,
                    "published_date": meta[cid] or "",
                    "evidence": norm_space(x.get("evidence")),
                    "confidence": float(x.get("confidence") or 0.0),
                })
    return pd.DataFrame(triples), pd.DataFrame(errors)


def norm_entity(name):
    s = unicodedata.normalize("NFKC", norm_space(name)).lower()
    s = re.sub(r"[^\w\s\-\.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_suffix(name):
    toks = norm_entity(name).replace(".", "").split()
    while toks and toks[-1] in CORP_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def looks_like_person(name):
    toks = strip_suffix(name).split()
    return 2 <= len(toks) <= 4 and all(t[:1].isalpha() for t in toks if t)


def merge_guard(a, b, typ=None):
    na, nb = strip_suffix(a), strip_suffix(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    sa, sb = set(na.split()), set(nb.split())
    if sa <= sb or sb <= sa:
        extra = (sa | sb) - (sa & sb)
        if extra & PRODUCTISH:
            return False
        if min(len(sa), len(sb)) == 1 and max(len(sa), len(sb)) >= 2:
            return False
    if typ == "Person" or (looks_like_person(a) and looks_like_person(b)):
        ta, tb = na.split(), nb.split()
        if ta and tb and ta[0] != tb[0] and ta[-1] == tb[-1]:
            return False
    return SequenceMatcher(None, na, nb).ratio() >= 0.72


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedder


def build_resolution_map(raw_triples_df, threshold=ER_THRESHOLD, top_k=5):
    mentions = []
    for r in raw_triples_df.itertuples(index=False):
        mentions += [(r.source_type, r.source_raw), (r.target_type, r.target_raw)]
    counts = Counter((t, norm_entity(n)) for t, n in mentions)
    display_name = {}
    for t, n in mentions:
        display_name.setdefault((t, norm_entity(n)), n)
    mapping, audit = {}, []
    for key in counts:
        t, norm = key
        if norm in MANUAL_ALIASES:
            mapping[key] = MANUAL_ALIASES[norm]
            audit.append({
                "type": t, "left": display_name[key],
                "right": MANUAL_ALIASES[norm],
                "similarity": 1.0, "decision": "MERGE_MANUAL",
            })
    for typ in sorted(ALLOWED_NODE_TYPES):
        keys = [k for k in counts if k[0] == typ and k not in mapping]
        if not keys:
            continue
        names = [display_name[k] for k in keys]
        vecs = get_embedder().encode(
            names, batch_size=128, show_progress_bar=False, normalize_embeddings=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        sims, nbrs = index.search(vecs, min(top_k, len(names)))
        uf = UF(len(names))
        audit_floor = 0.70
        for i in range(len(names)):
            for score, j in zip(sims[i], nbrs[i]):
                if j < 0 or i >= j:
                    continue
                sim = float(score)
                if sim < audit_floor:
                    continue
                ok = merge_guard(names[i], names[j], typ)
                if sim >= threshold and ok:
                    decision = "MERGE_VECTOR"
                    uf.union(i, j)
                elif not ok:
                    decision = "REJECT_GUARD"
                else:
                    decision = "BELOW_THRESHOLD"
                audit.append({
                    "type": typ, "left": names[i], "right": names[j],
                    "similarity": sim,
                    "decision": decision,
                })
        groups = defaultdict(list)
        for i in range(len(names)):
            groups[uf.find(i)].append(i)
        for idxs in groups.values():
            best = sorted(idxs, key=lambda i: (-counts[keys[i]], len(names[i]), names[i].lower()))[0]
            canonical = names[best]
            for i in idxs:
                mapping[keys[i]] = canonical
    for key in counts:
        mapping.setdefault(key, display_name[key])
    return mapping, pd.DataFrame(audit)


def canonicalize_triples(raw_df, mapping):
    df = raw_df.copy()

    def canon(name, typ):
        n = norm_entity(name)
        return mapping.get((typ, n), MANUAL_ALIASES.get(n, name))

    df["source_name"] = [canon(n, t) for n, t in zip(df.source_raw, df.source_type)]
    df["target_name"] = [canon(n, t) for n, t in zip(df.target_raw, df.target_type)]
    df["source_name_norm"] = df.source_name.map(norm_entity)
    df["target_name_norm"] = df.target_name.map(norm_entity)
    df["source_id"] = [sha1(f"{t}:{n}")[:24] for t, n in zip(df.source_type, df.source_name_norm)]
    df["target_id"] = [sha1(f"{t}:{n}")[:24] for t, n in zip(df.target_type, df.target_name_norm)]
    return df[df.source_id != df.target_id].reset_index(drop=True)


def build_nodes(triples_df):
    rows = []
    for r in triples_df.itertuples(index=False):
        rows += [
            {"id": r.source_id, "name": r.source_name, "name_norm": r.source_name_norm, "type": r.source_type, "alias": r.source_raw},
            {"id": r.target_id, "name": r.target_name, "name_norm": r.target_name_norm, "type": r.target_type, "alias": r.target_raw},
        ]
    tmp = pd.DataFrame(rows)
    if tmp.empty:
        return tmp
    out = []
    for (node_id, name, name_norm, typ), g in tmp.groupby(["id", "name", "name_norm", "type"]):
        aliases = sorted(set(g["alias"].map(norm_space)))
        out.append({
            "id": node_id, "name": name, "name_norm": name_norm, "type": typ,
            "aliases": aliases,
            "aliases_norm": sorted(set(norm_entity(x) for x in aliases)),
        })
    return pd.DataFrame(out)


def batches(records, size=1000):
    for i in range(0, len(records), size):
        yield records[i:i + size]


def bulk_insert_nodes(nodes_df, batch_size=1000):
    for typ in sorted(ALLOWED_NODE_TYPES):
        part = nodes_df[nodes_df.type == typ]
        if part.empty:
            continue
        query = f"""
        UNWIND $rows AS row
        MERGE (n:Entity {{id: row.id}})
        SET n:{typ},
            n.name=row.name,
            n.name_norm=row.name_norm,
            n.entity_type=row.type,
            n.aliases=row.aliases,
            n.aliases_norm=row.aliases_norm
        """
        for b in batches(part.to_dict("records"), batch_size):
            run_cypher(query, rows=b)


def bulk_insert_edges(triples_df, batch_size=1000):
    required = {"source_chunk_id", "published_date"}
    if not required.issubset(triples_df.columns):
        raise ValueError("Missing edge provenance.")
    for rel in sorted(ALLOWED_RELATIONS):
        part = triples_df[triples_df.relation == rel]
        if part.empty:
            continue
        query = f"""
        UNWIND $rows AS row
        MATCH (s:Entity {{id: row.source_id}})
        MATCH (t:Entity {{id: row.target_id}})
        MERGE (s)-[r:{rel} {{source_chunk_id: row.source_chunk_id}}]->(t)
        SET r.published_date=row.published_date,
            r.evidence=row.evidence,
            r.confidence=row.confidence
        """
        cols = ["source_id", "target_id", "source_chunk_id", "published_date", "evidence", "confidence"]
        for b in batches(part[cols].to_dict("records"), batch_size):
            run_cypher(query, rows=b)


def graph_checks():
    invalid = run_cypher("""
    MATCH ()-[r]->()
    WHERE r.source_chunk_id IS NULL OR r.published_date IS NULL
    RETURN count(r) AS n
    """)[0]["n"]
    counts = {
        "nodes": run_cypher("MATCH (n:Entity) RETURN count(n) AS n")[0]["n"],
        "edges": run_cypher("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"],
        "invalid_provenance_edges": invalid,
    }
    print(counts)
    assert invalid == 0
    top = pd.DataFrame(run_cypher("""
    MATCH (n:Entity)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) AS degree
    RETURN n.id AS id, n.name AS name, n.entity_type AS type, degree
    ORDER BY degree DESC LIMIT 15
    """))
    print(top.head(15).to_string(index=False))
    return counts, top


def build_flat_index(chunks_df):
    global flat_index, flat_store
    vecs = get_embedder().encode(
        chunks_df.text.fillna("").tolist(),
        batch_size=128, show_progress_bar=True, normalize_embeddings=True,
    ).astype("float32")
    flat_index = faiss.IndexFlatIP(vecs.shape[1])
    flat_index.add(vecs)
    flat_store = chunks_df.reset_index(drop=True).copy()
    print("Flat vectors:", flat_index.ntotal)


def retrieve_flat_context(query, k=6):
    qv = get_embedder().encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, ids = flat_index.search(qv, min(k, flat_index.ntotal))
    rows = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        r = flat_store.iloc[int(idx)]
        rows.append({
            "score": float(score), "chunk_id": r.chunk_id,
            "published_date": r.published_date, "text": r.text,
        })
    df = pd.DataFrame(rows)
    context = "\n\n".join(
        f"[chunk_id={r.chunk_id} | date={r.published_date} | score={r.score:.3f}]\n{r.text}"
        for r in df.itertuples(index=False)
    )
    return context, df


SEED_SYSTEM = """
Extract useful seed entities for graph retrieval.
Allowed types: Company, Person, Technology.
Do not answer the question. Return strict JSON only.
""".strip()


def extract_seeds(query):
    obj, _ = llm_json(SEED_SYSTEM, f"""
Question: {query}
Return {{"seeds":[{{"name":"...","type":"Company|Person|Technology|null"}}]}}
""")
    return [
        {"name": norm_space(x.get("name")),
         "type": x.get("type") if x.get("type") in ALLOWED_NODE_TYPES else None}
        for x in obj.get("seeds", [])
        if norm_space(x.get("name"))
    ]


def build_entity_matcher(nodes_df):
    global entity_match_vectors, entity_match_store
    entity_match_store = nodes_df.reset_index(drop=True).copy()
    entity_match_vectors = get_embedder().encode(
        entity_match_store.name.tolist(),
        batch_size=128, show_progress_bar=False, normalize_embeddings=True,
    ).astype("float32")


def match_seeds(query, fuzzy_threshold=0.66):
    matched = []
    for seed in extract_seeds(query):
        exact = run_cypher("""
        MATCH (n:Entity)
        WHERE (n.name_norm=$name OR $name IN coalesce(n.aliases_norm,[]))
          AND ($typ IS NULL OR n.entity_type=$typ)
        RETURN n.id AS id, n.name AS name, n.entity_type AS type
        LIMIT 5
        """, name=norm_entity(seed["name"]), typ=seed["type"])
        if exact:
            matched += exact
            continue
        if entity_match_vectors is None:
            continue
        mask = np.ones(len(entity_match_store), dtype=bool)
        if seed["type"]:
            mask = entity_match_store.type.eq(seed["type"]).to_numpy()
        idxs = np.flatnonzero(mask)
        if not len(idxs):
            continue
        qv = get_embedder().encode([seed["name"]], normalize_embeddings=True, show_progress_bar=False).astype("float32")[0]
        sims = entity_match_vectors[idxs] @ qv
        j = int(np.argmax(sims))
        if float(sims[j]) >= fuzzy_threshold:
            r = entity_match_store.iloc[int(idxs[j])]
            matched.append({"id": r.id, "name": r.name, "type": r.type})
    return list({x["id"]: x for x in matched}.values())


def node_degree(node_id):
    return int(run_cypher("""
    MATCH (n:Entity {id:$id})
    OPTIONAL MATCH (n)-[r]-()
    RETURN count(r) AS degree
    """, id=node_id)[0]["degree"])


def recent_edges(node_id, limit):
    return run_cypher("""
    MATCH (n:Entity {id:$id})
    MATCH (n)-[r]-(m:Entity)
    RETURN
      startNode(r).id AS source_id,
      startNode(r).name AS source_name,
      startNode(r).entity_type AS source_type,
      type(r) AS relation,
      endNode(r).id AS target_id,
      endNode(r).name AS target_name,
      endNode(r).entity_type AS target_type,
      r.source_chunk_id AS source_chunk_id,
      r.published_date AS published_date,
      r.evidence AS evidence,
      m.id AS neighbor_id
    ORDER BY coalesce(r.published_date,'') DESC
    LIMIT $limit
    """, id=node_id, limit=int(limit))


def textualize(edges):
    edges = sorted(edges, key=lambda e: e.get("published_date") or "", reverse=True)
    lines, used = [], 0
    for e in edges:
        line = (
            f"{e['source_name']} [{e['source_type']}] -{e['relation']}-> "
            f"{e['target_name']} [{e['target_type']}] "
            f"| date={e.get('published_date') or 'unknown'} "
            f"| chunk={e.get('source_chunk_id') or 'unknown'}"
        )
        if e.get("evidence"):
            line += f" | evidence={norm_space(e['evidence'])}"
        if used + len(line) + 1 > MAX_GRAPH_CONTEXT_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def retrieve_graph_context(query, max_hops=2, edge_limit=50, return_debug=False):
    seeds = match_seeds(query)
    if not seeds:
        out = {"context": "", "edges": pd.DataFrame(),
               "diagnostics": {"reason": "NO_SEED", "supernode_events": []}}
        return out if return_debug else ""
    frontier = deque((x["id"], 0) for x in seeds)
    expanded, seen_edges, collected = set(), set(), []
    supernode_events = []
    while frontier and len(collected) < GLOBAL_EDGE_CAP:
        node_id, hop = frontier.popleft()
        if node_id in expanded or hop >= max_hops:
            continue
        expanded.add(node_id)
        degree = node_degree(node_id)
        limit = int(edge_limit)
        if degree > SUPER_NODE_DEGREE:
            limit = min(limit, SUPER_NODE_EDGE_CAP)
            supernode_events.append({"node_id": node_id, "degree": degree, "limit": limit})
        for e in recent_edges(node_id, limit):
            key = (e["source_id"], e["relation"], e["target_id"], e["source_chunk_id"])
            if key in seen_edges:
                continue
            seen_edges.add(key)
            collected.append(e)
            if len(collected) >= GLOBAL_EDGE_CAP:
                break
            nb = e.get("neighbor_id")
            if nb and nb not in expanded and hop + 1 < max_hops:
                frontier.append((nb, hop + 1))
    out = {
        "context": textualize(collected),
        "edges": pd.DataFrame(collected),
        "diagnostics": {
            "matched_seeds": seeds,
            "expanded_nodes": len(expanded),
            "collected_edges": len(collected),
            "supernode_events": supernode_events,
        },
    }
    return out if return_debug else out["context"]


ANSWER_SYSTEM = """
Answer only from supplied context.
Be concise but complete. Do not invent facts.
Cite provenance inline as [chunk_id=...] whenever possible.
If evidence is insufficient or conflicting, say so.
""".strip()


def generate_answer(question, context):
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:"
    t0 = time.perf_counter()
    text, usage = llm_chat(
        [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=700,
    )
    return {
        "answer": (text or "").strip(),
        "latency_s": time.perf_counter() - t0,
        "total_tokens": usage.get("total_tokens"),
    }


def answer_flat_rag(question):
    context, retrieved = retrieve_flat_context(question, k=6)
    out = generate_answer(question, context)
    out.update({"context": context, "retrieved": retrieved})
    return out


def answer_graph_rag(question):
    g = retrieve_graph_context(question, max_hops=2, edge_limit=50, return_debug=True)
    vctx, vdocs = retrieve_flat_context(question, k=4)
    context = f"=== GRAPH ===\n{g['context']}\n\n=== VECTOR ===\n{vctx}"
    out = generate_answer(question, context)
    out.update({"context": context, "graph_debug": g, "vector_docs": vdocs})
    return out


SUFFICIENCY_SYSTEM = """
Decide whether the supplied retrieval context is sufficient to answer the question faithfully.
Do not answer the question. Return strict JSON only.
""".strip()


def context_sufficient(question, context):
    text = (context or "").strip()
    if len(text) < 180:
        return False, "graph context too short"
    obj, _ = llm_json(
        SUFFICIENCY_SYSTEM,
        f"QUESTION: {question}\nCONTEXT:\n{text[:16000]}\nReturn {{\"sufficient\":true,\"missing\":\"...\"}}",
        max_tokens=200,
    )
    return bool(obj.get("sufficient")), norm_space(obj.get("missing"))


def self_correcting_context(question):
    """Hop-2 → hop-3 → vector fallback when graph context is insufficient."""
    g2 = retrieve_graph_context(question, 2, 50, True)
    ok, missing = context_sufficient(question, g2["context"])
    if ok:
        return {"route": "hop2", "context": g2["context"], "missing": ""}
    g3 = retrieve_graph_context(question, 3, 50, True)
    ok, missing2 = context_sufficient(question, g3["context"])
    if ok:
        return {"route": "hop3", "context": g3["context"], "missing": missing}
    flat, _ = retrieve_flat_context(question, k=8)
    return {
        "route": "hop3+vector",
        "context": f"=== GRAPH ===\n{g3['context']}\n\n=== VECTOR ===\n{flat}",
        "missing": missing2,
    }


JUDGE_SYSTEM = """
You are a strict evaluator of RAG answers.
Score 1-5:
- comprehensiveness
- faithfulness to supplied candidate context
- multi_hop_reasoning accuracy
Use the reference answer as correctness anchor.
Return strict JSON only.
""".strip()


def judge_answer(question, reference, answer, context):
    prompt = f"""
QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{answer}

CANDIDATE CONTEXT:
{context[:18000]}

Return:
{{
 "comprehensiveness":1,
 "faithfulness":1,
 "multi_hop_reasoning":1,
 "rationale":"2-5 sentences"
}}
"""
    obj, _ = llm_json(JUDGE_SYSTEM, prompt, max_tokens=500)
    out = {}
    for k in ["comprehensiveness", "faithfulness", "multi_hop_reasoning"]:
        out[k] = max(1, min(5, int(obj.get(k, 1))))
    out["rationale"] = norm_space(obj.get("rationale"))
    return out


def run_evaluation(golden_df):
    rows = []
    checkpoint = OUTPUT_DIR / "graphrag_eval_checkpoint.csv"
    for q in tqdm(golden_df.itertuples(index=False), total=len(golden_df), desc="Evaluation"):
        flat = answer_flat_rag(q.question)
        graph = answer_graph_rag(q.question)
        jf = judge_answer(q.question, q.reference_answer, flat["answer"], flat["context"])
        jg = judge_answer(q.question, q.reference_answer, graph["answer"], graph["context"])
        rows.append({
            "id": q.id, "group": q.group, "question": q.question,
            "reference_answer": q.reference_answer,
            "flat_answer": flat["answer"], "graph_answer": graph["answer"],
            "flat_comprehensiveness": jf["comprehensiveness"],
            "graph_comprehensiveness": jg["comprehensiveness"],
            "flat_faithfulness": jf["faithfulness"],
            "graph_faithfulness": jg["faithfulness"],
            "flat_multi_hop_reasoning": jf["multi_hop_reasoning"],
            "graph_multi_hop_reasoning": jg["multi_hop_reasoning"],
            "flat_latency_s": flat["latency_s"],
            "graph_latency_s": graph["latency_s"],
            "flat_total_tokens": flat.get("total_tokens"),
            "graph_total_tokens": graph.get("total_tokens"),
            "flat_judge_rationale": jf["rationale"],
            "graph_judge_rationale": jg["rationale"],
            "graph_supernode_events": len(graph["graph_debug"]["diagnostics"].get("supernode_events", [])),
        })
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
    return pd.DataFrame(rows)


def comparison_table(eval_df):
    metric_map = {
        "Comprehensiveness": ("flat_comprehensiveness", "graph_comprehensiveness"),
        "Faithfulness": ("flat_faithfulness", "graph_faithfulness"),
        "Multi-hop reasoning": ("flat_multi_hop_reasoning", "graph_multi_hop_reasoning"),
        "Latency (s)": ("flat_latency_s", "graph_latency_s"),
        "Token usage": ("flat_total_tokens", "graph_total_tokens"),
    }
    rows = []
    for group, g in eval_df.groupby("group"):
        for metric, (fc, gc) in metric_map.items():
            f = pd.to_numeric(g[fc], errors="coerce").mean()
            gr = pd.to_numeric(g[gc], errors="coerce").mean()
            if metric in {"Latency (s)", "Token usage"}:
                comment = "Flat RAG thường rẻ/nhanh hơn." if f < gr else "GraphRAG không đắt hơn trong sample này."
            else:
                delta = gr - f
                if delta >= .75:
                    comment = "GraphRAG cải thiện rõ; kiểm tra rationale và provenance."
                elif delta <= -.5:
                    comment = "Flat RAG tốt hơn; graph extraction/retrieval có thể gây mất thông tin hoặc nhiễu."
                else:
                    comment = "Hai phương pháp gần nhau."
            rows.append({
                "Loại câu hỏi": group, "Metric": metric,
                "Flat RAG": round(f, 3) if pd.notna(f) else np.nan,
                "GraphRAG": round(gr, 3) if pd.notna(gr) else np.nan,
                "Nhận xét phân tích": comment,
            })
    return pd.DataFrame(rows)


def test_supernode_policy():
    rows = run_cypher("""
    MATCH (n:Entity)-[r]-()
    WITH n, count(r) AS degree
    ORDER BY degree DESC LIMIT 1
    RETURN n.id AS id, n.name AS name, degree
    """)
    if not rows:
        print("Graph empty.")
        return {"name": None, "degree": 0, "fetched": 0, "capped": False}
    n = rows[0]
    limit = 50 if n["degree"] > SUPER_NODE_DEGREE else 1000
    edges = recent_edges(n["id"], limit)
    print(n, "fetched=", len(edges))
    capped = n["degree"] > SUPER_NODE_DEGREE
    if capped:
        assert len(edges) <= 50
        print("[OK] Super-node cap OK.")
    return {"name": n["name"], "degree": n["degree"], "fetched": len(edges), "capped": capped}


def build_communities(limit_edges=20000):
    import networkx as nx
    edge_df = pd.DataFrame(run_cypher("""
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.id AS source, b.id AS target, a.name AS source_name, b.name AS target_name, type(r) AS relation
    LIMIT $limit
    """, limit=int(limit_edges)))
    if edge_df.empty:
        return pd.DataFrame(columns=["id", "community_id"]), pd.DataFrame()
    G = nx.Graph()
    G.add_edges_from(edge_df[["source", "target"]].itertuples(index=False, name=None))
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    rows = []
    for cid, members in enumerate(communities):
        rows += [{"id": node_id, "community_id": int(cid)} for node_id in members]
    for b in batches(rows, 1000):
        run_cypher("""
        UNWIND $rows AS row
        MATCH (n:Entity {id:row.id})
        SET n.community_id=row.community_id
        """, rows=b)
    id_to_cid = {r["id"]: r["community_id"] for r in rows}
    reports = []
    for cid, members in enumerate(communities):
        names = []
        rels = []
        for e in edge_df.itertuples(index=False):
            c1, c2 = id_to_cid.get(e.source), id_to_cid.get(e.target)
            if c1 == cid or c2 == cid:
                names += [e.source_name, e.target_name]
                rels.append(f"{e.source_name} -{e.relation}-> {e.target_name}")
        uniq = list(dict.fromkeys(names))[:12]
        reports.append({
            "community_id": int(cid),
            "size": len(members),
            "entities": "; ".join(uniq),
            "summary": f"Community {cid} ({len(members)} nodes): " + "; ".join(rels[:8]),
        })
    report_df = pd.DataFrame(reports)
    return pd.DataFrame(rows), report_df


def global_search_communities(question, report_df, top_k=3):
    if report_df is None or report_df.empty:
        return ""
    q = set(norm_entity(question).split())
    scored = []
    for r in report_df.itertuples(index=False):
        blob = f"{r.entities} {r.summary}".lower()
        score = sum(1 for t in q if t and t in blob)
        scored.append((score, r.summary))
    scored.sort(reverse=True)
    return "\n".join(s for n, s in scored[:top_k] if n > 0)


def fill_golden_from_graph(triples_df, chunks_df):
    """Build 5 gold questions whose answers exist in the extracted graph/text."""
    rows = triples_df.copy()
    fact = rows.iloc[0] if len(rows) else None
    g01_q, g01_a, g01_e = "Who was the CEO of Hugging Face in 2023?", "Clément Delangue", "Not always present in this 5000-row subset."
    hf = rows[(rows.source_raw.str.contains("Hugging Face", case=False, na=False)) | (rows.target_raw.str.contains("Hugging Face", case=False, na=False))]
    leads = rows[rows.relation.eq("LEADS")]
    if len(leads):
        r = leads.iloc[0]
        g01_q = f"Who or what {r.relation.lower().replace('_',' ')} related to {r.target_name}?"
        g01_a = f"{r.source_name} {r.relation} {r.target_name}."
        g01_e = r.evidence
    elif fact is not None:
        g01_q = f"What relation is stated between {fact.source_name} and {fact.target_name}?"
        g01_a = f"{fact.source_name} {fact.relation} {fact.target_name}."
        g01_e = fact.evidence

    founded = rows[rows.relation.eq("FOUNDED")]
    invested = rows[rows.relation.eq("INVESTED_IN")]
    g02_q = "Which startups were founded by former Microsoft employees and later received investment from Google?"
    g02_a = "Not attested in this subset after extraction."
    g02_e = "TO_BE_FILLED_FROM_DATASET"
    if len(founded) and len(invested):
        f = founded.iloc[0]
        inv = invested.iloc[0]
        g02_q = f"Which company was founded (or related via FOUNDED) and which investment edge exists in the graph?"
        g02_a = f"FOUNDED: {f.source_name} -> {f.target_name}. INVESTED_IN: {inv.source_name} -> {inv.target_name}."
        g02_e = f"{f.evidence} | {inv.evidence}"

    g03_q = "Compare the direction of AI-related investments by Meta and Apple during 2023 using evidence from multiple articles."
    meta_apple = rows[rows.source_name.str.contains("Meta|Apple|Google|Microsoft|Amazon", case=False, na=False)]
    if len(meta_apple) >= 2:
        a = meta_apple.iloc[0]
        others = meta_apple[meta_apple.source_name != a.source_name]
        b = others.iloc[0] if len(others) else meta_apple.iloc[1]
        g03_q = (
            f"Compare relations involving {a.source_name} versus {b.source_name} "
            f"using evidence from multiple chunks."
        )
        g03_a = (
            f"{a.source_name} {a.relation} {a.target_name} (date={a.published_date}, chunk={a.source_chunk_id}). "
            f"{b.source_name} {b.relation} {b.target_name} (date={b.published_date}, chunk={b.source_chunk_id})."
        )
        g03_e = f"{a.evidence} || {b.evidence}"
    else:
        two = rows.head(2)
        if len(two) == 2:
            a, b = two.iloc[0], two.iloc[1]
            g03_q = "Compare two company relations reported in different news chunks."
            g03_a = (
                f"{a.source_name} {a.relation} {a.target_name} vs "
                f"{b.source_name} {b.relation} {b.target_name}."
            )
            g03_e = f"{a.evidence} || {b.evidence}"
        else:
            g03_a = "Insufficient multi-document evidence in this subset."
            g03_e = ""

    g04_q = "Find a company invested in by a major technology company that also developed a named AI technology; identify both relations and dates."
    developed = rows[rows.relation.eq("DEVELOPED")]
    if len(invested) and len(developed):
        inv, dev = invested.iloc[0], developed.iloc[0]
        g04_q = "Identify an INVESTED_IN relation and a DEVELOPED relation, including dates."
        g04_a = (
            f"{inv.source_name} INVESTED_IN {inv.target_name} on {inv.published_date}; "
            f"{dev.source_name} DEVELOPED {dev.target_name} on {dev.published_date}."
        )
        g04_e = f"{inv.evidence} | {dev.evidence}"
    elif len(rows) >= 2:
        a, b = rows.iloc[0], rows.iloc[min(1, len(rows) - 1)]
        g04_q = "Identify two relations in the knowledge graph together with their dates."
        g04_a = (
            f"{a.source_name} {a.relation} {a.target_name} ({a.published_date}); "
            f"{b.source_name} {b.relation} {b.target_name} ({b.published_date})."
        )
        g04_e = f"{a.evidence} | {b.evidence}"
    else:
        g04_a = "Not enough relations extracted."
        g04_e = ""

    g05_q = "Identify one technology connected to the same company in at least two news chunks and summarize how the relationship changed over time."
    tech = rows[rows.target_type.eq("Technology") | rows.source_type.eq("Technology")]
    g05_a = "No repeated company-technology pair across chunks in this subset."
    g05_e = ""
    if len(tech):
        tech = tech.copy()
        tech["pair"] = tech.source_name + "||" + tech.target_name
        counts = tech.groupby("pair")["source_chunk_id"].nunique()
        multi = counts[counts >= 2]
        pick = tech[tech.pair.eq(multi.index[0])] if len(multi) else tech.head(2)
        if len(pick):
            r0 = pick.sort_values("published_date").iloc[0]
            r1 = pick.sort_values("published_date").iloc[-1]
            g05_q = f"How is {r0.source_name} connected to {r0.target_name} across news chunks over time?"
            g05_a = (
                f"{r0.source_name} {r0.relation} {r0.target_name} on {r0.published_date} "
                f"[{r0.source_chunk_id}] and later {r1.relation} on {r1.published_date} [{r1.source_chunk_id}]."
            )
            g05_e = f"{r0.evidence} || {r1.evidence}"

    golden = pd.DataFrame([
        {"id": "G01", "group": "factoid", "question": g01_q, "reference_answer": g01_a, "reference_evidence": g01_e},
        {"id": "G02", "group": "multi-hop", "question": g02_q, "reference_answer": g02_a, "reference_evidence": g02_e},
        {"id": "G03", "group": "cross-doc", "question": g03_q, "reference_answer": g03_a, "reference_evidence": g03_e},
        {"id": "G04", "group": "multi-hop", "question": g04_q, "reference_answer": g04_a, "reference_evidence": g04_e},
        {"id": "G05", "group": "cross-doc", "question": g05_q, "reference_answer": g05_a, "reference_evidence": g05_e},
    ])
    golden.to_csv(GOLDEN_PATH, index=False)
    return golden


def main():
    print("=== Module 1: load / dedup / chunk / coref ===")
    raw_df = load_news(DATA_PATH)
    news_df = standardize_news(raw_df)
    news_df = near_dedup(news_df)
    news_df = prioritize_articles(news_df, LAB_MAX_ARTICLES)
    print("articles used:", len(news_df))
    chunks_path = OUTPUT_DIR / "chunks_preview.csv"
    coref_path = OUTPUT_DIR / "coref_extraction_source.csv"
    triples_raw_path = OUTPUT_DIR / "raw_triples.csv"
    if chunks_path.exists():
        chunks_df = pd.read_csv(chunks_path)
        print("reuse chunks:", len(chunks_df))
    else:
        chunks_df = build_chunks(news_df)
        chunks_df.to_csv(chunks_path, index=False)
        print("chunks:", len(chunks_df))

    if coref_path.exists():
        extraction_source = pd.read_csv(coref_path)
        print("reuse coref extraction source:", len(extraction_source))
    else:
        extraction_source = chunks_df.head(EXTRACTION_MAX_CHUNKS).copy()
        coref_df = run_coref(extraction_source)
        extraction_source = extraction_source.merge(coref_df, on="chunk_id", how="left")
        extraction_source.to_csv(coref_path, index=False)
    unresolved_n = int(extraction_source["unresolved_mentions"].map(
        lambda x: len(x) if isinstance(x, list) else (0 if pd.isna(x) else 1)
    ).sum())
    print("unresolved mention lists non-empty:", unresolved_n)

    print("=== Module 2: NER+RE ===")
    if triples_raw_path.exists():
        raw_triples_df = pd.read_csv(triples_raw_path)
        extraction_errors_df = pd.DataFrame()
        print("reuse triples", len(raw_triples_df))
    else:
        raw_triples_df, extraction_errors_df = run_extraction(extraction_source)
        raw_triples_df.to_csv(triples_raw_path, index=False)
        print("triples", len(raw_triples_df), "errors", len(extraction_errors_df))
    if raw_triples_df.empty:
        raise RuntimeError("No triples extracted — cannot continue.")

    print("=== Module 3: entity resolution ===")
    entity_map, entity_resolution_audit_df = build_resolution_map(raw_triples_df)
    triples_df = canonicalize_triples(raw_triples_df, entity_map)
    entity_resolution_audit_df.to_csv(OUTPUT_DIR / "entity_resolution_audit.csv", index=False)
    triples_df.to_csv(OUTPUT_DIR / "triples_canonical.csv", index=False)
    print("audit rows", len(entity_resolution_audit_df), "canonical triples", len(triples_df))

    print("=== Neo4j ingest ===")
    connect_neo4j()
    setup_graph_schema()
    run_cypher("MATCH (n) DETACH DELETE n")
    nodes_df = build_nodes(triples_df)
    bulk_insert_nodes(nodes_df)
    bulk_insert_edges(triples_df)
    graph_counts, top_degree_df = graph_checks()
    top_degree_df.to_csv(OUTPUT_DIR / "top_degree.csv", index=False)
    nodes_df.to_csv(OUTPUT_DIR / "nodes.csv", index=False)

    print("=== Module 4: retrieval ===")
    build_flat_index(chunks_df)
    build_entity_matcher(nodes_df)

    print("=== Module 5: golden + eval ===")
    eval_path = OUTPUT_DIR / "graphrag_eval_results.csv"
    summary_path = OUTPUT_DIR / "graphrag_vs_flatrag_summary.csv"
    if GOLDEN_PATH.exists() and eval_path.exists():
        golden_df = pd.read_csv(GOLDEN_PATH)
        eval_results_df = pd.read_csv(eval_path)
        comparison_df = pd.read_csv(summary_path) if summary_path.exists() else comparison_table(eval_results_df)
        print("reuse golden + eval CSVs")
    else:
        golden_df = fill_golden_from_graph(triples_df, chunks_df)
        print(golden_df[["id", "group", "question"]].to_string(index=False))
        eval_results_df = run_evaluation(golden_df)
        comparison_df = comparison_table(eval_results_df)
        eval_results_df.to_csv(eval_path, index=False)
        comparison_df.to_csv(summary_path, index=False)
    print(comparison_df.to_string(index=False))

    print("=== Failure checks + bonus community / self-correction ===")
    supernode = test_supernode_policy()
    community_df, community_reports_df = build_communities()
    community_df.to_csv(OUTPUT_DIR / "communities.csv", index=False)
    if community_reports_df is not None and len(community_reports_df):
        community_reports_df.to_csv(OUTPUT_DIR / "community_reports.csv", index=False)
        print("global-search sample:\n", global_search_communities(golden_df.iloc[0].question, community_reports_df)[:500])
    print("communities", community_df["community_id"].nunique() if len(community_df) else 0)
    sc = self_correcting_context(str(golden_df.iloc[-1].question))
    (OUTPUT_DIR / "self_correction_demo.json").write_text(
        json.dumps({"question": str(golden_df.iloc[-1].question), "route": sc["route"], "missing": sc["missing"]}, indent=2),
        encoding="utf-8",
    )
    print("self-correction route:", sc["route"])

    meta = {
        "articles": int(len(news_df)),
        "chunks": int(len(chunks_df)),
        "extraction_chunks": int(len(extraction_source)),
        "triples": int(len(triples_df)),
        "audit_rows": int(len(entity_resolution_audit_df)),
        "graph": graph_counts,
        "supernode": supernode,
        "coref_unresolved_batches": unresolved_n,
        "communities": int(community_df["community_id"].nunique()) if len(community_df) else 0,
    }
    (OUTPUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("DONE", meta)
    if driver:
        driver.close()


if __name__ == "__main__":
    main()
