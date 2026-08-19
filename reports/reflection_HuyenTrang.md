# Reflection cá nhân — Lab 19 GraphRAG vs Flat RAG

**Học viên:** **Hoàng** Huyền Trang - 2A202601311  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày:** 19/08/2026  

---

## 1. Mapping bài giảng vào code


| Khái niệm trong bài giảng      | Module | Hàm / khối code                                       | Quan sát                                                   |
| ------------------------------ | ------ | ----------------------------------------------------- | ---------------------------------------------------------- |
| Conservative Coreference       | M1     | `resolve_coref_batch()`, `run_coref()`                | Ambiguous pronouns được log, không bịa antecedent          |
| Schema & Allowlist Guard       | M2     | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`             | Loại relation lạ trước Cypher                              |
| Bulk Cypher Ingestion          | M2     | `bulk_insert_nodes()`, `bulk_insert_edges()`          | `UNWIND $rows AS row`, batch 1000; 0 cạnh thiếu provenance |
| Entity Resolution & Union-Find | M3     | `build_resolution_map()`, `UF`, `merge_guard()`       | Threshold 0.90 + guard substring/person-name               |
| Super-node Degree Cap          | M4     | `retrieve_graph_context()`, `test_supernode_policy()` | Policy đúng; subset nhỏ chưa kích hoạt degree>100          |
| LLM-as-a-Judge                 | M5     | `judge_answer()`, `run_evaluation()`                  | 5 câu × 3 tiêu chí; 2 CSV eval                             |


Bonus: `near_dedup()` (SimHash LSH); `build_communities()` (55 community + `community_id` UNWIND + community reports); `self_correcting_context()` (hop2 → hop3 → vector fallback).

Pipeline local: `run_lab19.py` (cùng logic notebook, cache CSV để tránh gọi lại LLM).

---



## 2. Debugging & bài học

Lỗi khó nhất:

1. Dataset gốc HackerNoon gated, thiếu `HF_TOKEN` agree — chuyển stream public 5000 dòng `AIatMongoDB/tech-news-embeddings`.
2. `UnicodeEncodeError` khi `print` emoji trên Windows cp1252 **sau khi Neo4j đã connect** — tưởng pipeline “mất graph”.
3. OpenAI hết credit → NVIDIA NIM; JSON mode không ổn định nên parse khối `{...}`.

Cách xử lý: cache CSV sau coref/NER; bỏ emoji trên console; UNWIND ingest lại; audit floor 0.70 để ≥10 dòng minh bạch.

Bài học: GraphRAG không tự thắng Flat RAG nếu extraction thưa hoặc subgraph nhiễu. Cần audit ER, provenance 0-null, và đọc qualitative song song với điểm judge.

---



## 3. Action plan đồ án

- **Bài toán:** Tra cứu quan hệ công ty / tin M&A–đầu tư, hoặc đồ án nội bộ có thực thể–quan hệ rõ.
- **Có cần GraphRAG?** Single-hop trên đoạn ngắn → Flat RAG đủ. Câu “ai đầu tư X rồi X acquire Y” → Hybrid GraphRAG. Lab cho thấy GraphRAG fail nếu KG sparse.
- **Node / Relation:** `Company`, `Person`, `Technology` (`Entity`); `ACQUIRED`, `INVESTED_IN`, `FOUNDED`, `PARTNERED_WITH`, `USES`, `LEADS`, `DEVELOPED`, `WORKED_AT`.
- **ER & Super-node:** Manual ticker + ANN 0.90 + lexical guard; không merge product chứa brand; super-node lấy 50 cạnh mới nhất **lọc theo thời gian query**.

---



## Tự đánh giá


| Tiêu chí        | 1–5 | Ghi chú                                              |
| --------------- | --- | ---------------------------------------------------- |
| Hiểu GraphRAG   | 4   | E2E, đo noisy-subgraph và missing-path               |
| Kiểm soát agent | 4   | Từ chối O(N²) và full 350MB                          |
| Chất lượng KG   | 3   | 121 cạnh / 0 thiếu provenance; graph thưa            |
| Debug hệ thống  | 4   | Resume cache, encoding Windows, judge vs qualitative |


Môi trường: 5000 rows stream → exact dedup 4976→2287 → SimHash 2285; 1500 articles / 1500 chunks; extract 80 chunks. Neo4j local `bolt://localhost:7687`. Không hard-code API key.