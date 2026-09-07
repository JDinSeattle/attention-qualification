# Attention Qualification

[![CPU contracts](https://github.com/JDinSeattle/attention-qualification/actions/workflows/ci.yml/badge.svg)](https://github.com/JDinSeattle/attention-qualification/actions/workflows/ci.yml)

Independent numerical and memory-safety qualification of **FlashInfer FA2 paged causal attention with split-KV disabled** on an RTX 4090. The reference starts from logical dense K/V; the implementation receives shuffled physical pages with poisoned tails. Shared backend code cannot make both sides of the comparison silently agree.

**18 fixed-seed GPU cases**, each run in an isolated worker process, cover ragged request lengths, GQA, exact page boundaries, sliding windows, nonunit softmax scales, caller-owned output/LSE buffers and real dispatch. [Measured error distributions and test evidence](docs/RESULTS.md).

## Reproduce

Requires Linux, CUDA toolkit/Compute Sanitizer, a real supported NVIDIA GPU and uv:

```bash
bash scripts/reproduce.sh
```

The locked environment and FlashInfer JIT cache default to paths without spaces under `/tmp`. This matters: the original environment path exposed an upstream JIT quoting failure, preserved in exploratory logs. Initial execution compiles four FA2 specializations. CPU-only: install PyTorch, NumPy and pytest, then `python scripts/run.py --cpu-only`.

Generate a readable report: `python scripts/summarize.py results/<run-directory>`.

## Independent checks

- FP32 mathematical attention reference without FlashInfer or PyTorch SDPA; FP64 roundoff checks and analytic singleton/GQA/window tests.
- Right-aligned causal positions, contiguous GQA head grouping and base-2 LSE derived explicitly.
- Random physical page permutation and NaN-poisoned unused slots; output/LSE guard regions and pointer identity.
- Actual FA2 dispatch plus a separate CUDA profiler trace; no automatic backend selection.
- Mutations for head mapping, causal alignment, page-tail masking, scale and window. A reducer produces a minimal synthetic failing batch.
- Correctness-first microbenchmarking with raw samples and recorded test-reference overhead.

The [contract](docs/contract.md) excludes zero-length/all-mask cases, quantization and other backends. `sm_scale` is a softmax scale, not a quantization scale. This narrow v1 does not claim FP8/FP4 or Blackwell coverage. A passing corpus is not a new upstream bug: the [review packet](docs/UPSTREAM_REVIEW.md) is local, and no upstream PR acceptance is claimed. See [interview evidence](docs/INTERVIEW.md).

The automatic split-KV merge path triggered a duplicate CUDA kernel-registration diagnostic under memcheck. Process isolation did not remove it. Both failed runs are retained; the qualified variant explicitly sets `disable_split_kv=True`, with no sanitizer suppression or claim that the upstream merge path is fixed.
