# Measured qualification report

Run: `verified-fa2-nosplit-20260907`. Execution status: **passed**.

[All commands and exit codes](../results/verified-fa2-nosplit-20260907/execution.json) · [CPU test log](../results/verified-fa2-nosplit-20260907/cpu-tests.log) · [GPU correctness](../results/verified-fa2-nosplit-20260907/gpu-check.json) · [Raw timing](../results/verified-fa2-nosplit-20260907/benchmark.json)

Source digest: `c8e2d7bef7ed999a4974be477a3f3c3fc1d8ce991503f1619126ff45a73dba04`.

Hardware: RTX 4090 (sm_89), driver 595.84, native toolkit CUDA 13.2. Clocks unchanged; display GPU. Results apply to this run and workload only.

CPU: 34 tests. All command return codes are retained; missing prerequisites fail the reproduction driver.

18 GPU cases passed. Each case runs in an isolated process with disable_split_kv=True. The automatic split-KV path retains a reproduced merge-kernel registration diagnostic. Both independent output and base-2 LSE comparisons pass; guards and caller buffer identity pass.

| Metric | Maximum across corpus |
|---|---:|
| Output absolute error | 0.000951052 |
| LSE absolute error | 0.000293255 |

| FA2 case | Event p50 µs | p05–p95 µs |
|---|---:|---:|
| singleton | 3.99 | 3.99–10.14 |
| ragged-tails | 22.66 | 4.10–39.96 |
| window-zero | 4.10 | 4.10–37.49 |
| window-one | 25.29 | 5.02–39.42 |
| window-page | 4.71 | 4.61–17.80 |
| equal-heads | 4.10 | 4.02–4.12 |
| causal-prefill | 23.04 | 5.53–31.54 |
| page-one | 4.10 | 3.99–32.97 |
| longer-kv | 19.88 | 19.87–19.97 |

[Actual CUDA dispatch names](../results/verified-fa2-nosplit-20260907/dispatch-trace.json) · [Reduced synthetic mutation](../results/verified-fa2-nosplit-20260907/minimal-mutant.json)

Microbenchmarks quantify this backend only; no CPU-to-GPU speedup claim is made. CPU reference cost is recorded separately per correctness case. Quantization, all-mask rows and other backends remain outside v1. No upstream bug is asserted from a passing suite.

## Safety and reproducibility

- [gpu-memcheck log](../results/verified-fa2-nosplit-20260907/gpu-memcheck.log): exit 0.

The reproduction driver fails on missing GPU, failed checks, stale gates, sanitizer errors or subprocess failure. CPU CI is labeled separately. All performance comparisons retain failures and unsupported cases.
