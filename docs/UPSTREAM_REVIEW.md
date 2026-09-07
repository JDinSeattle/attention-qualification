# Local upstream review packet

Status: local validation packet; no issue, comment or PR was sent.

Target: FlashInfer 0.6.18.post1, FA2 paged causal attention, FP16, sm_89. `manifests/corpus.json` specifies every supported combination and exclusion. The reference does not call FlashInfer or PyTorch SDPA. The current corpus passes output and base-2 LSE checks, protected caller buffers, memcheck and actual CUDA dispatch verification; see the measured report for logs.

Qualification is per isolated worker process with `disable_split_kv=True`. The automatic split-KV path produced a duplicate `PersistentVariableLengthMergeStatesKernel` registration diagnostic in Compute Sanitizer while its numerical checks passed. Isolation reproduced the error, so the initial cross-module-only explanation was rejected. Both failures remain available; this is a reproducible integration diagnostic requiring separate upstream investigation, not an asserted attention arithmetic defect or a claimed fixed loader.

The most reusable review material is the logical-KV reference, permuted physical pages with poisoned tails, analytic window/GQA cases, and mutation tests. `results/.../minimal-mutant.json` is an intentionally corrupted head-mapping example; it is explicitly not a real upstream defect.

If a future version fails, retain the exact version/environment and seed, minimize batch members with `ddmin`, and add only the resulting narrow regression to the upstream test layout after confirming its API contract. A zero-length or all-mask behavior must not be called a bug based on this lab's deliberately excluded domain.
