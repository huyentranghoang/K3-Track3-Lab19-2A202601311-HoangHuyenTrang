# Thuyết minh kỹ thuật — Lab 19 GraphRAG vs Flat RAG

**Học viên:** Huyen Trang  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày:** 19/08/2026  

File này trả lời các câu bảo vệ kiến trúc (RUBRIC 4.1). Phân tích ca lỗi nằm ở `failure_analysis.md`. Reflection nằm ở `reflection_HuyenTrang.md`.

---

## 1. Coreference Resolution

**Tình huống:** `chunk_id=65c63febf187c085a866f9ff::c0000` — bài *Truescope Acquires US Firm Universal Information Services*. Prompt conservative **không** gán `us` / `we` cho một công ty cụ thể (`unresolved_mentions = ['us', 'we']`).

Cùng đoạn có Truescope (bên mua) và UIS (bên bị mua). Nếu LLM gán nhầm *the company / we* cho UIS thì sự kiện M&A đảo chiều.

**Hậu quả:** False coreference → false edge (ví dụ `UIS -ACQUIRED-> Truescope`). Pipeline bỏ qua khi mơ hồ: đánh đổi recall để giữ precision đồ thị.

Rủi ro thứ hai: bài *Adobe student receives national Information and Technology award* nói về **Adobe Middle School**, không phải Adobe Inc. Coref/NER gán “Adobe” thành công ty phần mềm sẽ tạo node/cạnh sai.

---

## 2. Entity Resolution Threshold & Lexical Guard

- **Ngưỡng cosine:** `threshold = 0.90`. Chỉ MERGE khi ANN ≥ 0.90 **và** lexical guard pass. Audit log thêm cặp ≥ 0.70.
- **Cặp bị Guard chặn (cosine > 0.85):** `Samsung` vs `Samsung Electronics` (≈ **0.860**, `REJECT_GUARD`).
- **Lý do:** Guard từ chối khi một tên là tập con token của tên kia và lệch số token (1 vs 2) — nhằm chặn *Apple* vs *Apple Watch*. Hệ quả: `Samsung` / `Samsung Electronics` chưa gộp (conservative, phân mảnh tập đoàn).

Cặp gần ngưỡng, không merge: `Google Cloud Platform` vs `Google Cloud` (≈ 0.892, `BELOW_THRESHOLD`). Merge đúng: `Amazon Web Services (AWS)` → `Amazon Web Services` (`MERGE_VECTOR`, ≈ 0.929); `Microsoft Corporation` → `Microsoft` (`MERGE_MANUAL`).

Bảng audit **15 dòng** trong `outputs/entity_resolution_audit.csv`.

---

## 3. Super-node Analysis

| Hạng | Tên | Type | Degree |
|------|-----|------|--------|
| 1 | Microsoft | Company | 13 |
| 2 | Google | Company | 7 |
| 3 | Amazon Web Services | Company | 5 |

Subset lab (`EXTRACTION_MAX_CHUNKS=80`) **không có node degree > 100**. `test_supernode_policy()` chạy: Microsoft degree=13, cap 50 không kích hoạt. Policy: `degree > 100` → `ORDER BY published_date DESC LIMIT 50`, `GLOBAL_EDGE_CAP=250`.

- **Ưu điểm:** Chặn bùng nổ context Google/Microsoft khi scale 350MB; ưu tiên tin mới; giảm token.
- **Rủi ro:** Câu hỏi lịch sử (M&A cũ) mất cạnh vì cắt theo ngày mới. Nên lọc theo khoảng thời gian của câu hỏi.

Provenance: **`invalid_provenance_edges = 0`**. Đồ thị: **170 nodes / 121 edges**.

---

## 4. Benchmark Quality vs Latency vs Token

| Tiêu chí | Flat RAG | GraphRAG | Δ | Nhận xét |
|----------|----------|----------|---|----------|
| Comprehensiveness (1–5) | 3.40 | 2.00 | −1.40 | Graph nhiễu / thiếu seed multi-hop |
| Faithfulness (1–5) | 3.40 | 2.60 | −0.80 | GraphRAG thêm cạnh không liên quan |
| Multi-hop Reasoning (1–5) | 3.40 | 1.80 | −1.60 | Multi-hop Dragos/SYNLAB không gần seed |
| Latency TB (s) | 6.46 | 6.95 | +0.49 | Seed LLM + BFS + vector |
| Token TB | 865 | 1691 | +826 | Hybrid GRAPH + VECTOR ~2× token |

Chi tiết 2 ca lỗi: xem `failure_analysis.md`. CSV: `outputs/graphrag_eval_results.csv`, `outputs/graphrag_vs_flatrag_summary.csv`.

Judge: NVIDIA NIM `meta/llama-3.1-8b-instruct` (OpenAI hết credit, không Groq key local) — điểm ồn; qualitative quan trọng hơn số tuyệt đối.

---

## 5. Trade-offs, Agent Control & Scale 350MB

- **Quality vs cost vs latency:** GraphRAG đắt hơn (~2× token, +0.5s) và trên 80-chunk **chưa thắng** quality. Lợi ích kỳ vọng khi multi-hop *có đường đi* trên KG. Flat RAG đủ factoid khi snippet ngắn. Bottleneck GraphRAG = LLM NER/RE, không phải Neo4j UNWIND.
- **Từ chối đề xuất agent:** Từ chối pairwise cosine O(N²) near-dedup 5000 dòng. Dùng **SimHash 64-bit + LSH** (hamming ≤ 3): 2,287 → 2,285 bài. Từ chối tải dump gated 350MB; stream 5000 dòng public `AIatMongoDB/tech-news-embeddings`.
- **Scale 350MB:** Queue extract async; chỉ chunk có tín hiệu acquire/invest/CEO; HNSW/blocking ER; community partition + global search; super-node cap theo thời gian query.
