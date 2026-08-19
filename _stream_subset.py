"""Stream at most 5000 rows of tech-news text. Skip embedding vectors."""
from __future__ import annotations

import csv
import os
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "hackernoon_subset.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
LIMIT_ROWS = 5000
KEEP_COLS = ["_id", "companyName", "companyUrl", "published_at", "url", "title", "description"]

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if OUT.exists() and OUT.stat().st_size > 10_000:
    print(f"Reuse existing {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
else:
    names = ["AIatMongoDB/tech-news-embeddings", "MongoDB/tech-news-embeddings"]
    last_err = None
    dataset = None
    used = None
    for name in names:
        try:
            dataset = load_dataset(name, split="train", streaming=True)
            used = name
            break
        except Exception as e:
            last_err = e
    if dataset is None:
        raise RuntimeError(last_err)

    print(f"Streaming from {used}, max {LIMIT_ROWS} rows (no embeddings).")
    n = 0
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = None
        for row in tqdm(dataset, total=LIMIT_ROWS, desc="rows"):
            rec = {k: ("" if row.get(k) is None else row.get(k)) for k in KEEP_COLS}
            rec["text"] = rec.get("description") or ""
            rec["id"] = rec.get("_id") or rec.get("url") or str(n)
            rec["published_date"] = rec.get("published_at") or ""
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
                writer.writeheader()
            writer.writerow(rec)
            n += 1
            if n >= LIMIT_ROWS:
                break
    print(f"Wrote {n} rows -> {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
