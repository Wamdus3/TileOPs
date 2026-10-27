# Ascend 950 FFT migration

This change adds `AscendFFTC2CKernel` to the existing `FFTC2CFwdOp` interface.
It is a complex64 implementation, with complex128 coverage still blocked. It
does not complete the ten-case FFT acceptance gate.

## Source and migration

The source is the unified FFT experiment in
`Wamdus3/tilelang`, artifact
`output/fft-flow-merged-20261027/source/solution.py`, SHA256
`4f0de99e01b745fdeceb92775e3edfb79f6836d6f2e08b76c22158533c75a615`.
That generated artifact was untracked: infrastructure commit
`da3abfff5aa1cef7a11b1d4736b7735f455d3ebe` does not itself contain it.
The content hash identifies the imported implementation. The source repository
and TileOPs use MIT licensing; no AscendC library code is imported.

The TileOPs starting point is `ascend950` at
`b84f999c9ec5d375bcedb62b44d79863f830924a`. The migration follows the FFT portion
of the 2026-10-27 SSD/FFT implementation plan and
[FFT issue 15](https://github.com/yyttt6/TileOPs/issues/15).

| Concern                                                       | Treatment                                                        |
| ------------------------------------------------------------- | ---------------------------------------------------------------- |
| Forward sign, unnormalized output and natural frequency order | Reused                                                           |
| Local radix-8/register fusion and paired global DIT           | Reused as one arithmetic flow                                    |
| Row packing                                                   | Reused geometry rule; hardware vector-core count replaces 72     |
| Public complex tensor and arbitrary leading dimensions        | Adapted to dense split-FP32 buffers and back                     |
| Noncontiguous and lazy-conjugate input                        | Materialized before viewing the real storage                     |
| Output ownership                                              | New interleaved storage; no input alias or in-place update       |
| Complex128                                                    | Missing FP64 arithmetic path; explicitly refused before building |
| Acceptance and numerical policy                               | Public TileOPs reference and thresholds retained                 |

No complete-kernel selection by shape is introduced. Every supported length uses
the same stages: bit-reversed gather, constant-root first radix-8, local fused
triples with a radix-2 remainder, paired global stages, and a final radix-2 stage
when needed. Short transforms finish in the local stage. Each block owns its
complete rows; global stages use the output buffers as scratch and require no
cross-block FFT synchronization.

Long transforms allocate four complex data banks and two root tables, totaling
196608 UB bytes for the imported 4096-element tile. The caller remains serial
across dependent stages; this change does not claim pipelined overlap.

The port also chooses a power-of-two VF width that divides the local butterfly
group count. Five-row packets consequently use 128 threads, avoiding the source
compiler's partial-lane barrier-hoisting warning at 256 threads. This changes
scheduling and still requires new public-API device validation.

## Public support boundary

The registered implementation serves complex64, power-of-two lengths 2 through
65536\. Leading batch axes are flattened inside the implementation and restored
on return. Empty batches return a new contiguous empty tensor. The public op
already handles length one as an out-of-place identity.

The public call includes resolving conjugation/layout, extracting contiguous
real and imaginary components, executing the split FFT, and interleaving its
outputs. These allocations and tensor operations belong to the complete callable
and must remain inside the public benchmark boundary.

`fft-validation.json` lists all ten cases exported through the public case
loader, their unchanged numerical policies and validation status. The two
complex128 cases remain in the manifest and are blocked, rather than cast to
complex64 or silently skipped. The eight complex64 cases require new public-API
device verification; the source experiment's receipts cannot replace it.

The source experiment used uniform `atol=1.25e-4, rtol=1e-4`. TileOPs instead uses
the existing `FFTWorkload.verification`: `atol=rtol=1e-4` below 32768 for complex64,
and its length-dependent absolute tolerance for long transforms. Neither the
workload nor its reference was changed.

## Verification

Use the maintainer's Python 3.12, PyTorch/torch-npu 2.13, official TileLang 0.1.15
and CANN environment. Run from this checkout with unique output directories:

```bash
python scripts/ci/ascend950.py --phase cpu --output ./results/fft-cpu-001
python scripts/benchmark_ascend.py --owner dev/fft --run-dir ./results/fft-verify-001 --op FFTC2CFwdOp --verify-only
python scripts/npu_guard.py run --owner dev/fft-tests -- python -m pytest tests/ops/test_fft.py --tileops-device=npu --tileops-target=detect -m smoke
```

The public benchmark command obtains the shared host guard before importing the
NPU runtime. Direct pytest, compilation probes and autotuning on an NPU host
also require that same site guard. The full ten-case verification remains a
failure until complex128 is implemented; a successful subset is not acceptance.

New regression tests exercise a strided transformed axis, a partial row packet,
an odd packet's VF synchronization, and the public dispatch refusal for
complex128. Existing tests cover lazy
conjugation, length one and multidimensional batches. No test changes the
workload's reference or numerical policy.

The local CPU/lowering checks use the existing development environment, which
differs from the required official runtime. Their exact status is retained in
`fft-validation.json`. Public device and performance receipts remain pending.

## FP64 capability reproduction

The local CANN 9.2.0 `dav-3510` compiler rejected this minimal arithmetic probe
with `double precision operation is not allowed in aicore function`. This is
development-toolchain evidence; repeat the probe in the official runtime before
making a claim about that runtime. It performs compilation only:

```python
import tilelang
import tilelang.ascend.language as T


@T.prim_func
def fp64_add(x: T.Tensor((32,), "float64"), y: T.Tensor((32,), "float64")):
    with T.Kernel(1):
        a = T.alloc_shared((32,), "float64")
        b = T.alloc_shared((32,), "float64")
        T.copy(x, a)
        with T.SimtVF(threads=32):
            for i in T.Parallel(32):
                b[i] = a[i] + T.float64(1)
        T.copy(b, y)


tilelang.compile(fp64_add, target="ascend", out_idx=[1])
```

## Exploratory source measurements

These are the source experiment's split-buffer kernel durations, not TileOPs
performance acceptance. The experiment ran on an Ascend950DT_9592 using the
development compiler, three calls per group, no warmup and no auxiliary cache
flush. All eight AscendC groups were rerun using the library built from
CANN ops-fft commit `b8c6ed3f17201a4f3f19bcd5997b48cadd90782f`. Its CPU plan/root
preparation and the public layout adapter are excluded.

| Batch × length | Unified kernel median µs | AscendC median µs |
| -------------- | -----------------------: | ----------------: |
| 1 × 4096       |                   19.157 |           250.211 |
| 64 × 4096      |                   22.649 |           253.724 |
| 64 × 16384     |                  106.872 |          1344.707 |
| 8 × 65536      |                  427.083 |          6064.186 |
| 256 × 1024     |                   20.088 |           206.306 |
| 4096 × 512     |                  167.908 |          1537.053 |
| 65536 × 64     |                  254.018 |          2118.243 |
| 300 × 1024     |                   24.548 |           250.831 |

The unified kernel's geometric mean speedup was 1.1235× against the original
harness route and 1.0675× against the original direct route. The 64×16384 case
regressed 8.3% against direct. Three samples do not establish significance.

Formal performance remains pending the shared protocol and comparable baselines
for all agreed cases. The source experiment's Task Duration medians are not the
required 15-session raw-profile interval-union arithmetic means. No formal
performance score or complete correctness receipt is claimed by this commit.
