# GPU-Accelerated Sparse Direct Solvers: Literature Synthesis

## 1. Algorithmic Approaches

### Supernodal vs. Multifrontal

Sparse direct solvers achieve high performance by identifying dense submatrices within the sparse structure and operating on them with optimized BLAS routines. The two dominant paradigms are **supernodal** and **multifrontal** methods.

**Supernodal methods** (CHOLMOD, PaStiX, SuperLU, symPACK) operate directly on columns of the factor matrix grouped into "supernodes" -- contiguous sets of columns with identical sparsity structure. Updates are applied directly to the target columns in the factor storage. This avoids allocating temporary frontal matrices, reducing memory overhead. CHOLMOD and SuperLU are the canonical examples.

**Multifrontal methods** (MUMPS, STRUMPACK, SSIDS) organize computation around an assembly tree of dense "frontal matrices." Each frontal matrix is factored independently, producing a contribution block ("update matrix") that is assembled into its parent. The advantage is that each frontal matrix is a self-contained dense problem, naturally mapping to GPU BLAS calls. The disadvantage is higher temporary memory usage for the contribution blocks. STRUMPACK and SSIDS are the primary GPU-capable multifrontal solvers.

**Key tradeoff for GPU:** Multifrontal methods expose more parallelism per operation (each frontal matrix is independent), making them slightly more natural for GPU offloading. Supernodal methods have lower memory overhead and better locality, but require more careful scheduling to exploit GPU parallelism. In practice, both approaches achieve similar GPU speedups (3-5x) when well-implemented.

### Left-Looking vs. Right-Looking vs. Fan-Out

**Left-looking (LL):** When processing supernode j, all updates *from* previously-factored supernodes *to* j are gathered and applied before factoring j. CHOLMOD uses this approach. The advantage is data locality -- all reads target already-computed data. The disadvantage is limited parallelism: updates to j must be serialized. On GPU, the CHOLMOD "subtree" algorithm overcomes this by streaming entire subtrees of the elimination tree through GPU memory, performing the left-looking factorization entirely on-device for each subtree.

**Right-looking (RL):** When processing supernode j, its factorization immediately pushes updates *to* all descendant supernodes. This exposes more parallelism (many independent updates can proceed simultaneously) but has worse locality and requires synchronization. The 2024 paper by Sid-Lakhdar et al. (arXiv:2409.14009) implements two right-looking GPU variants:
- **RL:** Single large DSYRK call per supernode update. Simpler, better GPU utilization, but requires temporary storage for the full update matrix.
- **RLB (Right-Looking Blocked):** Multiple smaller DGEMM/DSYRK calls per block. Lower memory footprint but more kernel launches.

Their results show RL is generally faster on GPU (up to 4.47x speedup), except when the update matrix exceeds GPU memory, where RLB becomes necessary.

**Fan-out:** Used by symPACK (SC'23). Similar to right-looking but specifically designed for distributed memory with one-sided communication (UPC++). The factored supernode sends its contribution to all ancestors. symPACK's GPU implementation achieved up to 14x speedup over PaStiX on the Perlmutter supercomputer, demonstrating the effectiveness of this approach when combined with careful device offloading heuristics.

---

## 2. CPU vs. GPU Work Partitioning Strategies

This is perhaps the most critical design decision. Every system reviewed uses a **hybrid approach** -- the question is where to draw the line.

### Threshold-Based Offloading (Most Common)

**CHOLMOD (Rennich, Stosic, Davis 2014, 2016):** The foundational approach. Supernodes below a certain size are factored on CPU; above the threshold, BLAS operations are offloaded to GPU. Testing on the 100 largest SPD matrices from the SuiteSparse collection showed beneficial GPU acceleration for any factorization taking longer than 0.28 seconds on CPU. Maximum speedup was 3.5x, average 2.4x.

**arXiv:2409.14009 (Sid-Lakhdar et al. 2024):** Empirically determined thresholds of 600,000 nonzeros (for RL) and 750,000 nonzeros (for RLB) per supernode. Below threshold: CPU. Above: GPU. Quote: "for each supernode we check its size and if it is below a threshold, we keep it and all the computation associated with it on CPU." This is because "even if the computation in the GPU is much faster, for small volume of data the total transfer and computation time on GPU becomes more than the computation time on CPU."

**STRUMPACK (Ghysels et al. 2022):** For the multifrontal algorithm, large frontal matrices use vendor BLAS (cuBLAS/cuSOLVER for NVIDIA, rocBLAS/rocSOLVER for AMD). Small frontal matrices use custom CUDA/HIP kernels. Profiling showed that cuBLAS kernel launch overhead was prohibitively costly for small matrices, so they implemented a threshold: only call vendor BLAS for matrices above a certain size, and use fused custom kernels below it.

**symPACK (2023):** Uses a device offloading heuristic that considers both supernode size and the current GPU workload. The heuristic accounts for "memory kinds" -- a PGAS abstraction for managing host vs. device memory placement.

### Task-Based Scheduling (PaStiX)

**PaStiX** takes a different approach by using the **StarPU** runtime system for heterogeneous task scheduling. Rather than a fixed threshold, StarPU maintains a DAG (directed acyclic graph) of tasks and dynamically assigns them to CPU or GPU based on:
- Task granularity (size of the dense operation)
- Current device load
- Data locality (where the input data currently resides)
- A "fitness score" for each task/worker combination

This is more adaptive but adds runtime overhead. PaStiX with StarPU uses a **Heteroprio** scheduler that automatically assigns priorities to each task type.

### GPU-Resident Approaches (cuDSS, SuperLU_DIST batched)

**NVIDIA cuDSS** represents the newer trend: keep the entire factorization on GPU when possible. CPU handles only reordering/symbolic analysis. The factorization and solve phases run entirely on GPU. When factors exceed GPU memory, cuDSS offers a "hybrid memory mode" that keeps factors in host memory and streams portions to GPU as needed.

**SuperLU_DIST batched solver (2024):** For batched problems (many small-to-medium systems), the entire sparse LU factorization runs on GPU, computed level-by-level through the elimination tree. Each level launches batched GPU kernels for diagonal factorization, panel factorization, and Schur complement updates.

---

## 3. GPU Scheduling and Batching

### The Core Problem

Sparse factorization generates thousands of small-to-medium dense BLAS operations. Individually, these are too small to saturate a GPU. Kernel launch overhead (typically 5-10 microseconds per launch) dominates for small operations.

### Batching Strategies

**CHOLMOD subtree batching:** The "subtree algorithm" (introduced in CHOLMOD v4.6.0-beta) batches BLAS operations across an entire subtree of the elimination tree. An entire branch is loaded into GPU memory, and all BLAS kernels within that subtree are launched in batches, reducing both PCIe transfer overhead and kernel launch latency.

**SuperLU_DIST level-set batching (2024):** At each level of the elimination tree, all independent supernodes are processed as a batch. Three batched GPU kernels are launched per level:
1. Batched diagonal factorization (batched GETRF)
2. Batched panel factorization (batched TRSM)
3. Batched Schur complement update (batched GEMM)
A new "batched Scatter GPU kernel" handles the sparse assembly between levels. On A100, this is orders of magnitude faster than batched banded solvers.

**STRUMPACK custom kernels:** Rather than batching through vendor APIs, STRUMPACK implements hand-written CUDA/HIP kernels that process many small frontal matrices in a single kernel launch. Each thread block handles one small frontal matrix. This avoids the overhead of cuBLAS entirely for small operations.

**GLU 3.0 three-mode kernels:** GLU 3.0 dynamically selects between three GPU kernel modes depending on the factorization stage:
- Early stages: many independent columns, use one mode
- Middle stages: moderate parallelism, different mode
- Late stages: few columns with high dependency, third mode
Each mode adapts GPU block/warp allocation to match the available parallelism, dynamically balancing computing demands with resources.

### Key Insight

The consensus across all systems is that **batching is essential**. The specific batching granularity varies:
- CHOLMOD: batch across subtrees
- SuperLU_DIST: batch across elimination tree levels
- STRUMPACK: batch across small frontal matrices in custom kernels
- GLU 3.0: batch across columns within a level, with adaptive kernel modes

---

## 4. Pivoting Strategies on GPU

Pivoting is the most architecturally disruptive operation for GPU sparse solvers because it introduces data-dependent memory access patterns and cross-thread communication.

### No Pivoting (Cholesky)

For SPD matrices, Cholesky factorization requires no pivoting. This is why GPU-accelerated Cholesky (CHOLMOD, symPACK, the 2024 arXiv paper) achieves the best speedups -- the factorization pattern is entirely determined by the sparsity structure, known before numeric factorization begins.

### Static Pivoting (SuperLU_DIST)

SuperLU_DIST uses a **"perturb then refine"** strategy:
1. Before factorization, compute a column permutation (using MC64 or similar) to place large entries on the diagonal
2. During factorization, if a tiny pivot is encountered, replace it with sqrt(epsilon) * ||A|| -- a small perturbation
3. After factorization, perform iterative refinement to recover accuracy

This completely eliminates runtime pivoting decisions, making the factorization pattern static and GPU-friendly. The pivot/permutation sequence is computed once on CPU during symbolic analysis and reused for all subsequent factorizations.

### Threshold Pivoting (SSIDS, Dense LU)

**SSIDS** implements threshold partial pivoting within its multifrontal GPU solver. Rather than selecting the largest pivot (full partial pivoting), it accepts any pivot whose magnitude exceeds a threshold fraction of the column maximum. This reduces the number of row swaps and the associated irregular memory accesses. Research shows threshold pivoting improves GPU performance by up to 32% without significant accuracy loss. With more aggressive thresholds (allowing up to one digit of accuracy loss), speedups reach 44%.

### Perturbation-Induced Static Pivoting (Emerging)

A 2023 paper (arXiv:2311.11833) proposes a novel approach: instead of pivoting, solve a series of perturbed well-conditioned systems in parallel on GPU, then linearly combine results to recover the true solution. This converts the pivoting problem into a batch-solving problem, naturally suited to GPU parallelism.

### Practical Consensus

For GPU sparse LU, the field has largely converged on **static pivoting** (SuperLU_DIST) or **threshold pivoting** (SSIDS) as the best tradeoffs between numerical stability and GPU efficiency. Full partial pivoting is considered too expensive on GPU due to the irregular memory access patterns it induces.

---

## 5. Memory Management

### GPU Memory as the Primary Constraint

GPU memory (16-80 GB on current hardware) is typically much smaller than host memory, and sparse factor matrices can be very large. Every system must address this.

**CHOLMOD subtree approach:** Load one subtree at a time into GPU memory. Process it entirely on GPU, then replace with the next subtree. This naturally bounds GPU memory usage to the largest subtree. The algorithm also minimizes PCIe transfers by keeping the entire subtree resident.

**arXiv:2409.14009 RL vs. RLB tradeoff:** RL requires preallocating temporary storage for the largest update matrix on both CPU and GPU. For the matrix nlpkkt120, this exceeded the 40 GB A100 memory. RLB avoids this by computing and transferring back each small update matrix immediately, reducing peak GPU memory. RLB successfully factorized nlpkkt120 where RL could not.

**STRUMPACK sub-tree splitting:** The multifrontal algorithm identifies independent sub-trees whose factorization fits entirely in GPU memory. Each sub-tree is processed sequentially on GPU. This is a natural partitioning for memory management.

**cuDSS hybrid memory mode:** When factors exceed GPU memory, cuDSS keeps the full L and U factors in host memory and streams portions to GPU as needed. This adds host-device transfer cost but removes the memory limit. An automatic memory selection heuristic chooses between full-GPU and hybrid modes.

**SuperLU_DIST distributed:** Uses MPI to distribute the problem across multiple GPU nodes, with each node handling a portion of the elimination tree. This scales GPU memory linearly with the number of nodes.

### Supernode Merging

Several systems (including the 2024 arXiv paper) merge small supernodes before GPU factorization, allowing up to 25% fill increase. This reduces the total number of supernodes (fewer kernel launches) and increases the average supernode size (better GPU utilization), at the cost of some additional computation on zero fill-in entries.

---

## 6. Key Performance Insights

### Bottlenecks

1. **Small supernodes dominate real problems.** Most supernodes in practical sparse matrices are small (< 100 columns). These are terrible for GPU utilization individually. Batching, custom kernels, or CPU fallback are essential.

2. **PCIe transfer overhead.** Moving data between host and device costs 10-15 GB/s on PCIe Gen4. For small supernodes, transfer time exceeds compute time. Solutions: subtree algorithms (CHOLMOD), unified memory (Apple Silicon), GPU-resident approaches (cuDSS).

3. **Kernel launch overhead.** Each CUDA kernel launch costs 5-10 microseconds. For thousands of small BLAS operations, this adds up. Solutions: batching, custom fused kernels, or persistent kernels.

4. **Symbolic analysis is CPU-bound.** Reordering (AMD, METIS), symbolic factorization, and elimination tree construction all run on CPU. This is typically 10-30% of total time and does not benefit from GPU.

5. **Memory bandwidth, not compute, is the bottleneck.** Sparse factorization has low arithmetic intensity compared to dense BLAS. The scatter/gather operations between supernodes are memory-bound. Custom scatter kernels on GPU help but do not eliminate this bottleneck.

### What Optimizations Matter Most

1. **Supernode aggregation/merging** -- increasing average supernode size by 2-3x dramatically improves GPU utilization
2. **Batched operations** -- processing many small operations in a single kernel launch
3. **Overlapping compute and transfer** -- asynchronous PCIe transfers, CUDA streams, concurrent CPU/GPU execution
4. **Custom kernels for small operations** -- avoiding vendor BLAS overhead for matrices < 64x64
5. **Elimination tree scheduling** -- processing independent subtrees in parallel, choosing the right traversal order

### Speedup Summary

| System | Algorithm | GPU Speedup vs CPU | Hardware | Year |
|--------|-----------|-------------------|----------|------|
| CHOLMOD | Left-looking supernodal | 2.4x avg, 3.5x max | NVIDIA K40 | 2014-2016 |
| SSIDS | Multifrontal | up to 4.6x | NVIDIA GPU | 2016 |
| STRUMPACK | Multifrontal | 5x avg vs SuperLU | NVIDIA V100 | 2022 |
| symPACK | Fan-out supernodal | up to 14x vs PaStiX | NVIDIA A100 (Perlmutter) | 2023 |
| GLU 3.0 | Level-based LU | 6.7x geomean vs GLU 2.0 | NVIDIA GPU | 2019 |
| arXiv:2409.14009 | Right-looking supernodal | 1.3-4.5x | NVIDIA A100 | 2024 |
| cuDSS | Hybrid | 8x vs CPU (COMSOL) | NVIDIA H100 | 2024-2025 |
| SuperLU_DIST batched | Level-based LU | orders of magnitude vs banded | NVIDIA A100 | 2024 |

---

## 7. Per-System Detailed Summary

### CHOLMOD (SuiteSparse)
- **Algorithm:** Left-looking supernodal Cholesky
- **CPU vs GPU:** Threshold-based. Small supernodes on CPU, large on GPU. Subtree algorithm streams entire elimination tree branches through GPU.
- **Small supernodes:** Kept on CPU. Batched BLAS calls reduce launch overhead for GPU operations.
- **Pivoting:** None (Cholesky, SPD matrices only)
- **Key numbers:** 2.4x average, 3.5x maximum speedup. Beneficial when CPU factorization > 0.28 seconds.

### SuperLU_DIST
- **Algorithm:** Right-looking supernodal LU with static pivoting
- **CPU vs GPU:** CPU handles symbolic analysis, reordering, static pivot computation. GPU handles numeric factorization via level-set batched operations.
- **Small supernodes:** Batched GPU kernels process all supernodes at each elimination tree level together.
- **Pivoting:** Static pivoting with diagonal perturbation. Tiny pivots replaced with sqrt(eps)*||A||, followed by iterative refinement.
- **Key numbers:** Batched solver orders of magnitude faster than banded solver on A100.

### GLU 3.0
- **Algorithm:** Level-based parallel sparse LU for circuit simulation
- **CPU vs GPU:** CPU performs symbolic analysis and data dependency detection. GPU performs numeric LU factorization with three adaptive kernel modes.
- **Small supernodes:** Three kernel modes adapt to varying parallelism at different stages.
- **Pivoting:** Uses pre-computed pivot ordering from symbolic phase.
- **Key numbers:** 6.7x geometric mean speedup over GLU 2.0.

### STRUMPACK
- **Algorithm:** Multifrontal LU
- **CPU vs GPU:** Large frontal matrices use vendor BLAS on GPU. Small frontal matrices use custom CUDA/HIP kernels. Scatter-gather operations run on GPU.
- **Small supernodes:** Hand-written device kernels process small frontal matrices, avoiding cuBLAS overhead.
- **Pivoting:** Partial pivoting within frontal matrices.
- **Key numbers:** 5x average vs SuperLU on V100. 1.9x faster than cuDSS on average for SuiteSparse matrices.

### SSIDS (SPRAL)
- **Algorithm:** Multifrontal LDLT for symmetric indefinite systems
- **CPU vs GPU:** Both factorize and solve phases on GPU. Uses OpenMP tasks for scheduling.
- **Small supernodes:** Handled within the multifrontal framework.
- **Pivoting:** Threshold partial pivoting -- accepts pivots exceeding a fraction of column maximum.
- **Key numbers:** Up to 4.6x vs CPU multifrontal solver.

### symPACK
- **Algorithm:** Fan-out supernodal Cholesky
- **CPU vs GPU:** Device offloading heuristic based on supernode size and GPU workload. Uses UPC++ for distributed memory with one-sided communication.
- **Small supernodes:** Offloading heuristic keeps small blocks on CPU.
- **Pivoting:** None (Cholesky, SPD only)
- **Key numbers:** Up to 14x vs PaStiX on Perlmutter (A100 GPUs).

### PaStiX
- **Algorithm:** Left-looking supernodal (originally), now supports runtime-based scheduling
- **CPU vs GPU:** Uses StarPU/PaRSEC runtime systems for dynamic task-to-device assignment. Fitness-score-based scheduling via Heteroprio.
- **Small supernodes:** Runtime dynamically decides based on task granularity and device load.
- **Pivoting:** Supports both Cholesky (no pivoting) and LU with pivoting.
- **Key numbers:** Outperformed by symPACK on GPU systems.

### cuDSS (NVIDIA)
- **Algorithm:** Not publicly documented in detail; believed to be supernodal/multifrontal hybrid
- **CPU vs GPU:** CPU handles reordering and symbolic analysis. GPU handles factorization and solve. Hybrid memory mode streams factors from host when GPU memory is insufficient.
- **Small supernodes:** Internal batching mechanisms.
- **Pivoting:** Supports LU, Cholesky, and LDLT factorizations. Uses static pivoting with scaling.
- **Key numbers:** 8x speedup in COMSOL on H100 vs Intel i9. STRUMPACK reports being 1.9x faster than cuDSS on average.

### MUMPS
- **Algorithm:** Multifrontal LU/LDLT
- **CPU vs GPU:** As of the latest literature review, MUMPS does **not** have native GPU acceleration. It runs on CPU only, though it can operate on systems with GPUs present.
- **Key note:** MUMPS remains competitive on CPU due to its highly optimized multifrontal implementation, but lacks GPU offloading that competitors now provide.

---

## 8. Relevance to BaSpaCho

BaSpaCho implements a batched supernodal Cholesky/LU solver with both CUDA and Metal backends. Several findings from this literature review are directly relevant:

1. **Threshold-based offloading** is the dominant strategy. BaSpaCho currently sends everything to GPU regardless of size -- adding a size threshold for CPU fallback on small lumps would match the field consensus.

2. **Batched kernel launches** are critical. BaSpaCho's recent work on fusing saveGemm dispatches into batched kernels for Metal aligns with the CHOLMOD subtree and SuperLU_DIST level-set approaches.

3. **Custom kernels for small operations** (as in STRUMPACK) are more effective than calling vendor BLAS for small supernodes. BaSpaCho's Metal backend with custom .metal kernels follows this pattern.

4. **Static pivoting** for LU is the practical choice on GPU. The "perturb then refine" approach from SuperLU_DIST is well-established and would eliminate BaSpaCho's per-lump D->H pivot transfer.

5. **Apple Silicon's unified memory** eliminates the PCIe bottleneck that dominates all the NVIDIA-based systems reviewed. This is a significant architectural advantage for BaSpaCho's Metal backend -- no data transfer overhead means even small supernodes can benefit from GPU acceleration.

6. **Right-looking approaches** generally achieve better GPU utilization than left-looking, at the cost of higher temporary memory usage. The RLB variant from the 2024 arXiv paper offers a middle ground.

---

## Sources

- [GPU Accelerated Sparse Cholesky Factorization (arXiv:2409.14009, 2024)](https://arxiv.org/html/2409.14009v1)
- [GLU3.0: Fast GPU-based Parallel Sparse LU Factorization (arXiv:1908.00204)](https://arxiv.org/abs/1908.00204)
- [Accelerating Sparse Cholesky Factorization on GPUs (Rennich, Stosic, Davis)](https://people.engr.tamu.edu/davis/publications_files/IA3_2014_Workshop_Rennich_Stosic_Davis_preprint.pdf)
- [High Performance Sparse Multifrontal Solvers on Modern GPUs (STRUMPACK)](https://www.sciencedirect.com/science/article/abs/pii/S0167819122000059)
- [symPACK: A GPU-Capable Fan-Out Sparse Cholesky Solver (SC'23)](https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624600)
- [A Sparse Symmetric Indefinite Direct Solver for GPU Architectures (SSIDS)](https://dl.acm.org/doi/10.1145/2756548)
- [Batched Sparse Direct Solver Design in SuperLU_DIST (2024)](https://journals.sagepub.com/doi/abs/10.1177/10943420241268200)
- [NVIDIA cuDSS Documentation](https://docs.nvidia.com/cuda/cudss/)
- [NVIDIA cuDSS Blog: Advances in Solver Technologies](https://developer.nvidia.com/blog/nvidia-cudss-advances-solver-technologies-for-engineering-and-scientific-computing/)
- [Solving Large-Scale Linear Sparse Problems with cuDSS](https://developer.nvidia.com/blog/solving-large-scale-linear-sparse-problems-with-nvidia-cudss)
- [Towards Perturbation-Induced Static Pivoting (arXiv:2311.11833)](https://arxiv.org/html/2311.11833)
- [Making Sparse Gaussian Elimination Scalable by Static Pivoting (Li, Demmel)](https://portal.nersc.gov/project/sparse/xiaoye-web/SC98/sc98.pdf)
- [Threshold Pivoting for Dense LU Factorization](https://www.netlib.org/utk/people/JackDongarra/PAPERS/Threshold_Pivoting_for_Dense_LU_Factorization.pdf)
- [PaStiX with StarPU Runtime Scheduling](https://inria.hal.science/hal-00925017)
- [Tim Davis Survey of Direct Methods for Sparse Linear Systems](https://people.engr.tamu.edu/davis/publications_files/survey_tech_report.pdf)
- [Newly Released Capabilities in SuperLU_DIST (ACM TOMS)](https://dl.acm.org/doi/abs/10.1145/3577197)
- [Algorithm 887: CHOLMOD Supernodal Sparse Cholesky](https://dl.acm.org/doi/10.1145/1391989.1391995)
