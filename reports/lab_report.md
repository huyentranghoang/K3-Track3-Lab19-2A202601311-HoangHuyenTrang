# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Huyen Trang  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026  

Bản tổng hợp. Nộp theo RUBRIC: `technical_defense.md`, `failure_analysis.md`, `reflection_HuyenTrang.md`.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ từ dữ liệu:** `chunk_id=65c63febf187c085a866f9ff::c0000` — bài *Truescope Acquires US Firm Universal Information Services*. Prompt conservative **không** gán `us` / `we` cho một công ty cụ thể (`unresolved_mentions = ['us', 'we']`).
- **Hiện tượng:** Cùng đoạn còn có Truescope (bên mua) và UIS (bên bị mua). Nếu LLM gán nhầm *the company / we* cho UIS thì sự kiện M&A sẽ đảo chiều.
- **Hậu quả đối với Graph:** False coreference → false edge (ví dụ `UIS -ACQUIRED-> Truescope` thay vì chiều đúng). Pipeline cố ý **bỏ qua** khi mơ hồ, đánh đổi recall để giữ precision của đồ thị.

Một rủi ro thứ hai: bài *Adobe student receives national Information and Technology award* nói về **Adobe Middle School**, không phải Adobe Inc. Nếu coref/NER gán “Adobe” thành công ty phần mềm sẽ tạo node/cạnh sai hoàn toàn.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` (chỉ MERGE khi ANN ≥ 0.90 **và** lexical guard pass). Audit log thêm các cặp ≥ 0.70 để minh bạch.
- **Cặp thực thể bị Guard chặn:** `Samsung` vs `Samsung Electronics` (cosine ≈ **0.860**, `REJECT_GUARD`).
- **Lý do chặn:** Guard từ chối khi một tên là **tập con token** của tên kia và số token lệch (1 vs 2). Quy tắc này nhằm chặn *Apple* vs *Apple Watch*. Hệ quả: `Samsung` / `Samsung Electronics` chưa được gộp — conservative, tránh false merge nhưng làm phân mảnh node cùng một tập đoàn.

Cặp gần ngưỡng nhưng **không** merge vì dưới 0.90: `Google Cloud Platform` vs `Google Cloud` (≈ 0.892, `BELOW_THRESHOLD`). Cặp được merge đúng: `Amazon Web Services (AWS)` → `Amazon Web Services` (`MERGE_VECTOR`, ≈ 0.929); `Microsoft Corporation` → `Microsoft` (`MERGE_MANUAL`).

Bảng audit có **15 dòng** (`MERGE_MANUAL`, `MERGE_VECTOR`, `REJECT_GUARD`, `BELOW_THRESHOLD`) trong `outputs/entity_resolution_audit.csv`.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top 3 Super-nodes (trên subset lab):**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | Microsoft | Company | 13 |
| 2 | Google | Company | 7 |
| 3 | Amazon Web Services | Company | 5 |

Với `LAB_MAX_ARTICLES=1500` và chỉ **80 chunks** đưa vào NER+RE, **không có node nào degree > 100**. Hàm `test_supernode_policy()` vẫn chạy: node cao nhất (Microsoft, degree=13) lấy đủ 13 cạnh, cap 50 không kích hoạt. Policy đã cài: `degree > 100` → `ORDER BY published_date DESC LIMIT 50`, cộng `GLOBAL_EDGE_CAP=250`.

- **Ưu điểm & Rủi ro của Temporal Mitigation:**
  - *Ưu điểm:* Chặn bùng nổ context ở Google/Microsoft khi scale 350MB; ưu tiên tin mới, giảm token generator.
  - *Rủi ro:* Câu hỏi lịch sử (M&A 2021, vòng gọi vốn cũ) có thể mất cạnh vì bị cắt theo ngày mới. Cần filter theo khoảng thời gian của câu hỏi, không chỉ “50 cạnh mới nhất”.

Sanity check provenance: **`invalid_provenance_edges = 0`** (mọi cạnh có `source_chunk_id` và `published_date`). Đồ thị: **170 nodes / 121 edges**.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, trung bình 5 câu):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 3.40 | 2.00 | −1.40 | Graph context nhiễu / thiếu seed trên multi-hop |
| **Faithfulness (1–5)** | 3.40 | 2.60 | −0.80 | GraphRAG thêm cạnh không liên quan câu hỏi |
| **Multi-hop Reasoning (1–5)** | 3.40 | 1.80 | −1.60 | Golden multi-hop (Dragos/SYNLAB) không nằm gần seed retrieval |
| **Latency trung bình (s)** | 6.46 | 6.95 | +0.49 | GraphRAG chậm hơn vì seed LLM + BFS + vector |
| **Token usage trung bình** | 865 | 1691 | +826 | Hybrid `=== GRAPH ===` + `=== VECTOR ===` gần gấp đôi token |

Judge dùng cùng overlay OpenAI-compatible (NVIDIA NIM `meta/llama-3.1-8b-instruct`) vì OpenAI hết credit và không có Groq key trên máy local. Điểm judge vì thế **ồn**; phân tích qualitative bên dưới quan trọng hơn số tuyệt đối.

#### Phân tích 2 Ca lỗi Điển hình:
1. **Ca GraphRAG trích đúng fact nhưng bị Judge trừ (G01, factoid):**
   - *Question ID & Câu hỏi:* G01 — *Who or what leads related to Artificial Intelligence?*
   - *Gold:* `Microsoft LEADS Artificial Intelligence` (evidence KPMG–Microsoft AI partnership).
   - *GraphRAG đã giải quyết như thế nào?* Traversal lấy đúng cạnh `Microsoft -LEADS-> Artificial Intelligence` kèm provenance chunk.
   - *Vì sao điểm Comprehensiveness thấp (2 vs 5)?* Hybrid context nhồi thêm Google/OpenAI/Docker → câu trả lời dài, loãng; judge 8B xem đó là thiếu tập trung. Đây là failure mode **context explosion / noisy subgraph**, dù chưa phải super-node degree>100.
2. **Ca GraphRAG thất bại cross-doc (G05):**
   - *Question ID & Câu hỏi:* G05 — *How is Intel connected to Intel Foundry Services across news chunks over time?*
   - *Flat RAG:* 5/5 (các chunk Intel–Synopsys–IFS nằm gần nhau về embedding).
   - *GraphRAG:* 1/1 — seed/traversal không ghép đủ hai chunk `2023-08-14` (DEVELOPED) và `2023-08-15` (USES).
   - *Nguyên nhân:* Entity “Intel Foundry Services” bị gán type `Technology` một nơi và quan hệ rời; hybrid vẫn bị graph block lấn át vector. Super-node cap không phải nguyên nhân (degree Intel = 4).
   - *Đề xuất khắc phục:* Query-time hop expansion (bonus self-correction), ưu tiên cạnh đúng seed pair, và **không** prepend toàn bộ subgraph nếu vector đã cover.

G02/G04 (multi-hop Dragos FOUNDED + SYNLAB INVESTED_IN) **cả hai hệ đều 1 điểm**: hai sự kiện không có đường đi chung trên đồ thị và không cùng neighborhood vector — đúng bản chất “multi-hop giả” khi extraction subset chỉ 80 snippets.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:**

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:** GraphRAG đắt hơn (~2× token, +0.5s latency) và trên subset 80-chunk **chưa thắng** quality. Lợi ích kỳ vọng nằm ở multi-hop *có* đường đi trên KG (ACQUIRED → INVESTED_IN cùng thực thể). Flat RAG đủ cho factoid khi chunk ngắn (~200 ký tự/snippet). Indexing overhead GraphRAG = LLM NER/RE (bottleneck) + UNWIND + FAISS; Flat RAG chỉ FAISS MiniLM.
- **Quyết định từ chối AI Coding Agent:** Từ chối **pairwise cosine O(N²)** cho near-dedup toàn bộ 5000 dòng. Thay bằng **SimHash 64-bit + banding LSH** (hamming ≤ 3): 2,287 → 2,285 bài, không OOM. Cũng từ chối tải full dump 350MB/gated `HackerNoon/tech-company-news-data-dump` (cần HF_TOKEN agree); chỉ stream **5000 dòng** bản public `AIatMongoDB/tech-news-embeddings`, bỏ cột embedding.
- **Giải pháp scale 350MB:** Bottleneck đầu tiên là **LLM extraction** (rate-limit + chi phí), không phải Neo4j UNWIND. Hướng xử lý: queue batch async, chỉ extract chunk có tín hiệu quan hệ (acquire/invest/CEO), HNSW/blocking ER, community partition + global search, super-node cap theo thời gian câu hỏi.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | Ambiguous pronouns được log, không bịa antecedent |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Loại relation lạ (ví dụ LEVERAGED) trước khi vào Cypher |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `UNWIND $rows AS row`, batch 1000; 0 cạnh thiếu provenance |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF`, `merge_guard()` | Threshold 0.90 + guard substring/person-name |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `test_supernode_policy()` | Policy đúng; subset nhỏ chưa kích hoạt degree>100 |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `run_evaluation()` | 5 câu × 3 tiêu chí; export 2 CSV |

Near-dedup bonus: `near_dedup()` (SimHash). Community bonus: `build_communities()` — **55** community, ghi `community_id` bằng UNWIND.

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** (1) Dataset gốc gated, không có `HF_TOKEN` — chuyển sang stream public 5000 dòng. (2) `UnicodeEncodeError` khi `print("✅")` trên console Windows cp1252, **sau khi Neo4j đã connect** — pipeline tưởng như “mất graph”. (3) OpenAI hết credit → fallback NVIDIA NIM, JSON mode không ổn định nên parse khối `{...}`.
- **Cách xử lý:** Cache CSV sau coref/NER để không gọi lại LLM; bỏ emoji; `UNWIND` ingest lại; audit floor 0.70 để có ≥10 dòng minh bạch.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Tra cứu quan hệ công ty công nghệ / tin M&A–đầu tư (cùng domain lab) hoặc đồ án nội bộ có thực thể–quan hệ rõ.
- **Đặc thù bài toán & Lý do chọn giải pháp:** Nếu câu hỏi chủ yếu single-hop trên đoạn văn ngắn → **Flat RAG đủ**. Nếu cần “ai đầu tư X rồi X acquire Y” → **Hybrid GraphRAG**. Lab này cho thấy GraphRAG **không tự thắng** nếu extraction sparse.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Company`, `Person`, `Technology` (`Entity`)
  - Relations: `ACQUIRED`, `INVESTED_IN`, `FOUNDED`, `PARTNERED_WITH`, `USES`, `LEADS`, `DEVELOPED`, `WORKED_AT`
- **Chiến lược xử lý Super-node & Entity Resolution:** Manual ticker map + ANN 0.90 + lexical guard; super-node lấy 50 cạnh mới nhất **có filter thời gian theo query**; không merge product chứa brand (`Apple Watch`).

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Chạy E2E, đo được noisy-subgraph và missing-path |
| Khả năng kiểm soát AI Coding Agent | 4 | Từ chối O(N²) và full 350MB download |
| Chất lượng đồ thị tri thức xây dựng | 3 | 121 cạnh / 0 thiếu provenance; graph còn thưa |
| Khả năng phân tích và debug hệ thống | 4 | Resume cache, Windows encoding, judge vs qualitative |

---

## Phụ lục — Scale & môi trường lab
- Data: 5000 rows stream, exact dedup 4976→2287, SimHash 2285, dùng 1500 articles / 1500 chunks, extract 80 chunks.
- Neo4j local Docker `bolt://localhost:7687`, schema constraint `Entity.id`.
- Không hard-code API key trong notebook.
