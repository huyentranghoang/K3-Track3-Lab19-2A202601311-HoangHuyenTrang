# Phân tích ca lỗi — Flat RAG vs GraphRAG

**Học viên:** Huyen Trang  
**Nguồn điểm:** `outputs/graphrag_eval_results.csv` (LLM-as-a-Judge, 5 câu G01–G05)

---

## Ca 1 — GraphRAG lấy đúng fact nhưng bị trừ điểm (G01, factoid)

- **Câu hỏi:** Who or what leads related to Artificial Intelligence?
- **Gold:** `Microsoft LEADS Artificial Intelligence` (evidence KPMG–Microsoft AI partnership, chunk `65c64500f187c085a867092b::c0000`).
- **Flat RAG:** Comprehensiveness/Faithfulness/Multi-hop = 5/5/5. Trả lời liệt kê nhiều thực thể từ vector, **không** nêu Microsoft LEADS AI.
- **GraphRAG:** 2/5/1. Traversal **có** cạnh `Microsoft -LEADS-> Artificial Intelligence` kèm provenance.

**Root cause:** Không phải thiếu cạnh, mà **noisy subgraph / context explosion**. Hybrid prepend toàn bộ láng giềng Google/OpenAI/Docker. Generator trả lời dài, loãng; judge 8B xem là thiếu tập trung. Degree Microsoft chỉ 13 — **chưa** phải super-node >100, nhưng failure mode cùng họ: quá nhiều cạnh không liên quan query.

**Khắc phục:** Ưu tiên cạnh khớp seed pair; không prepend full subgraph nếu vector đã cover; self-correction chỉ expand hop khi context **thiếu**, không khi đã đủ nhưng nhiễu.

---

## Ca 2 — GraphRAG thất bại cross-doc (G05)

- **Câu hỏi:** How is Intel connected to Intel Foundry Services across news chunks over time?
- **Gold:** Intel DEVELOPED IFS ngày 2023-08-14 (`...f6e3`) rồi USES 2023-08-15 (`...f6f3`).
- **Flat RAG:** 5/5/5 — hai chunk Intel–Synopsys–IFS gần nhau về embedding.
- **GraphRAG:** 1/1/1 — seed/traversal không ghép đủ hai quan hệ theo thời gian.

**Root cause:** “Intel Foundry Services” bị type `Technology`; hai quan hệ DEVELOPED/USES rời trên KG. Block `=== GRAPH ===` lấn át vector dù vector đã có đủ hai chunk. Super-node cap **không** phải nguyên nhân (degree Intel ≈ 4).

**Khắc phục:** Query-time hop 2→3 + vector fallback (`self_correcting_context`); rank cạnh theo cặp seed; trộn vector trước khi graph nếu graph ngắn hoặc lệch topic.

---

## Ca phụ — Multi-hop giả (G02 / G04)

Cả hai hệ **1 điểm**. Gold ghép `Dragos FOUNDED Partner Program` với `SYNLAB INVESTED_IN Florence` — **không có đường đi chung** trên đồ thị và không cùng neighborhood vector. Đây là giới hạn extraction 80 snippet, không phải bug traversal.

Multi-hop thật trong graph (ví dụ IWS `ACQUIRED` Hyperion `DEVELOPED` ICS) nên dùng cho round đánh giá sau.

---

## Tóm tắt so sánh

| | Flat RAG thắng khi | GraphRAG thắng khi | Cả hai fail khi |
|--|-------------------|--------------------|-----------------|
| | Chunk gần nhau về semantic (G05, G03) | Cạnh đúng nằm trên seed (G01 có cạnh, nhưng nhiễu làm mất điểm) | Hai sự kiện không linked (G02/G04) |

Chi tiết số liệu Quality / Latency / Token: `technical_defense.md` mục 4 và `outputs/graphrag_vs_flatrag_summary.csv`.
