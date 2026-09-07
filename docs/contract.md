# Paged attention qualification contract

Pinned implementation: FlashInfer 0.6.18.post1, `BatchPrefillWithPagedKVCacheWrapper`, explicitly `backend='fa2'` and `disable_split_kv=True`, NHD paged KV, FP16 Q/K/V/output, FP32 QK/softmax, causal attention. The [pinned implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18.post1/flashinfer/prefill.py) and installed source were checked; this does not depend on an evolving default backend.

For request b, local query i attends to logical key j when `j <= kv_len[b] - qo_len[b] + i`. With nonnegative `window_left`, also require `j >= kv_len[b] - qo_len[b] + i - window_left`. Window zero therefore selects exactly the aligned current key. GQA query head h maps to KV head `h // (num_q_heads / num_kv_heads)`.

The reference uses dense **logical** K/V lists. Only the adapter uses randomly permuted **physical** pages. Invalid page-tail storage contains NaN so accidental reads cannot plausibly pass as zero padding. Query, key and value values are first rounded to their actual FP16 representation, then promoted to FP32 for reference computation.

`scores = Q @ K.T * sm_scale`; output is `softmax(masked scores) @ V`. No FlashInfer or PyTorch SDPA is called by the reference. Analytic singleton checks establish the output and scalar dot-product LSE, and FP64 checks quantify reference roundoff. This API/version returns base-2 LSE, so the reference divides natural `logsumexp` by `ln(2)`.

| Boundary | v1 policy |
|---|---|
| Lengths | Nonempty batch, `1 <= qo_len <= kv_len` |
| Heads | Integral GQA grouping, equal heads included |
| Head dimension/page size | D=64/128, page size=1/16 |
| Sliding window | -1 (full causal), 0, 1, 3, 16 |
| Softmax scale | Positive nonunit values in corpus |
| Padding | Ignored; poison values in unused physical page positions |
| Output allocation | Caller supplies output and LSE buffers; pointer identity and guards checked |
| Empty/all-mask cases | Excluded and rejected before dispatch; no invented upstream semantics |
| Quantization | FP8/FP4 and quantization scales excluded; sm_scale is a softmax scale |
| Other features | No custom mask, RoPE, ragged API, FA3 or Blackwell backend claim |

Output tolerance is 0.003 + 0.004 * abs(reference); LSE tolerance is 0.002 + 0.002 * abs(reference). Every record retains max, p99 and RMS errors. Mutations deliberately corrupt GQA mapping, causal alignment, tail masking, softmax scale and sliding window. All must be detected. The delta-debugging example minimizes an injected mutation, not a claimed upstream bug.

`_backend` and the built callable are recorded, and a separate CUDA profiler pass verifies an actual prefill kernel name. Memcheck and profiler runs are separate from microbenchmarking. CPU tests never qualify FA2 by themselves.

The first native JIT attempt failed because the environment lived in a path containing spaces. The logs are retained. The reproduction environment and JIT cache now live in paths without spaces; package source is unmodified. Four backend specializations (D=64/128, window on/off) compile on sm_89. This practical build limitation is independent of attention numerical correctness.

Memcheck reported duplicate CUDA entry registration for `PersistentVariableLengthMergeStatesKernel` in the automatic split-KV merge path. Numerical comparisons still passed, but the reproduction driver rejected that run as a clean sanitizer qualification. The complete failing log remains under `results/verified-final-20260907`. A second experiment isolated each case into a separate process and reproduced the same diagnostic (`results/verified-isolated-20260907`), refuting the initial cross-module-only explanation. The qualified v1 variant explicitly disables split-KV; the failing automatic path remains excluded. Cases still use individual worker processes and Compute Sanitizer follows all children. No errors are suppressed, and no upstream fix is claimed.

## Upstream boundary

No upstream defect is asserted from a passing corpus. No issue, comment, or PR is submitted automatically. The tests, pinned corpus, independent reference and minimized mutation fixture are available for reviewers to inspect. The [contribution guide](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18.post1/CONTRIBUTING.md) remains the entry point if a genuine failure is later found.
