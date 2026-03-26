# Metal vs CUDA Performance Analysis

## Summary

The Metal backend is slower than CUDA/BLAS for supernodal Cholesky factorization due to **synchronization granularity** — Metal requires explicit CPU→GPU round-trips, while CUDA uses stream-ordered async dispatch with no CPU blocking. However, deferred synchronization and command buffer batching significantly reduce this gap.

## Benchmark Data (Apple M4 Pro, Feb 2026)

### After Sync Optimization (Feb 28, 2026)

Deferred synchronization reduces CPU→GPU round-trips from ~2N+3 per lump (where N = number of boards) to 1 per lump.

#### MERI n=4 (1500 params, medium problem)

| Solver | Before | After | Speedup |
|--------|--------|-------|---------|
| BLAS (16 threads) | 73ms | 80ms | - |
| Metal single | 693ms | 510ms | **1.36x** |
| Metal batch=16 | 477ms | 471ms | ~1.0x |

#### MERI n=7 (1500 params, large problem)

| Solver | Before | After | Speedup |
|--------|--------|-------|---------|
| BLAS (16 threads) | 121ms | 126ms | - |
| Metal single | 1052ms | 791ms | **1.33x** |
| Metal batch=16 | 715ms | 708ms | ~1.0x |

#### c6288_jacobian (small sparse matrix)

| Solver | Factor Time | vs BLAS |
|--------|-----------|---------|
| BLAS (16 threads) | 2.4ms | 1.0x |
| Metal single | 8.8ms | 3.7x slower |
| Metal batch=4 | 4.0ms | 1.7x slower |
| Metal batch=16 | 2.6ms | 1.1x slower |

## Root Cause: Synchronization Granularity

### CUDA execution model (async, stream-ordered)

```
for each lump:
  cusolverDnSpotrf(...)    // async — queued on GPU stream
  cublasStrsm(...)         // async — queued after potrf
  cublasSgemm(...)         // async — queued after trsm
  assemble_kernel<<<>>>    // async — queued after gemm
```

Every call is fire-and-forget. CUDA streams provide implicit ordering. The CPU races ahead enqueueing work. Only one synchronization at the very end.

### Metal execution model (after optimization)

```
for each lump:
  prepareAssemble(l)
    commitAndWait()          // CPU WAITS — only sync point per lump

  for each board:
    saveSyrkGemm:
      commitPending()        // Submit previous assemble, NO WAIT
      [mpsGemm encode...]   // Encode into pendingCmdBuf_
    assemble:
      encodeKernel(...)      // Same command buffer as GEMM

  factorLump:
    potrf: [mpsChol encode...]  // Encode into pendingCmdBuf_, NO WAIT
    trsm:  [mpsTrsm encode...]  // Same command buffer as potrf
           commitPending()      // Submit, NO WAIT
```

This gives **1 CPU wait per lump** (in prepareAssemble) instead of the previous 2N+3 waits per lump (where N = number of boards per lump). Metal command queue ordering guarantees sequential execution, so GPU→GPU dependencies are handled implicitly.

### Before optimization (for reference)

```
for each lump:
  commitAndWait()      // Wait 1
  for each board:
    commitAndWait()    // Wait 2 (flush assemble)
    [mpsGemm + wait]   // Wait 3 (GEMM)
  [mpsChol + wait]     // Wait 4 (potrf)
  [mpsTrsm + wait]     // Wait 5 (trsm)
```

## Why Batching Helps

Batching amortizes fixed per-commit overhead. Instead of 1 potrf per commit, N batch items share one commit. The batched path was already efficient (1 commit per kernel × batch), so the deferred sync optimization has minimal effect on batched performance.

## Remaining Bottlenecks

1. **MPS dispatch overhead**: Each MPS operation creates and ends its own compute encoder internally (~4-10μs per call). For many small lumps, this adds up.

2. **Kernel launch overhead**: Even with deferred sync, each command buffer submission has overhead (~2-5μs).

3. **prepareAssemble sync**: The one remaining wait per lump exists because `devSpanToChainOffset` (shared memory buffer) must be updated via CPU memcpy. Double-buffering could eliminate this.

4. **Single-precision only**: Metal lacks FP64 support, limiting accuracy for ill-conditioned problems.

## Potential Further Improvements

1. **Double-buffer devSpanToChainOffset** — Eliminate the last per-lump sync by ping-ponging between two buffers.

2. **Raise MPS thresholds** — Push more work through the fused `factor_lumps_kernel` which does potrf+trsm in a single kernel dispatch with zero sync overhead.

3. **MTLEvent pipelining** — Use GPU-side events to overlap lump execution without any CPU waiting.

4. **Batch MPS across lumps** — For independent small lumps, batch their MPS calls into one command buffer.
