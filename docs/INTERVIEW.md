# Interview and résumé evidence

Suggested résumé bullet:

> Qualified FlashInfer FA2 paged attention with split-KV disabled on a real RTX 4090 using an independent FP32 logical-KV reference; covered 18 seeded GQA/page/window cases, output/LSE buffers and CUDA dispatch, and validated the detector with semantic mutations and a batch reducer.

This supports inference-kernel validation and ML-infrastructure correctness roles. Do not claim FlashAttention implementation authorship, quantization coverage, an upstream bug or an accepted contribution.

Interview demonstrations:

1. Derive `kv_len - qo_len + query_index` and show why a top-left causal triangle is wrong for append attention.
2. Draw physical pages in shuffled order; explain why the oracle never reads those pages.
3. Derive head grouping and base-2 LSE. Run the constant-value GQA and window-zero analytic tests.
4. Corrupt head mapping and show a detector failure, then reduce the batch to one request.
5. Show an actual prefill CUDA kernel in the dispatch trace, not merely a wrapper class name.

All-mask/empty-row semantics and FP8 scale reconstruction need separate qualification. The present contract excludes them visibly.
