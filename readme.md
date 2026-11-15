# AVX2 $\pi$ Approximation Benchmark

A C++ implementation and performance benchmark comparing a standard scalar approach against a vectorized **Intel AVX2** implementation for approximating $\pi$ via numerical integration (Midpoint Riemann Sum).

## Mathematical Background

The value of $\pi$ is computed by numerically integrating the following definite integral over the interval $[0, 1]$:

$$\pi = \int_{0}^{1} \frac{4}{1 + x^2} \, dx$$

The interval is divided into $N$ subintervals (`STEPS`) of width $\Delta x = \frac{1}{N}$, evaluating the function at the midpoint of each rectangle:

$$\pi \approx \sum_{i=0}^{N-1} \frac{4}{1 + \left((i + 0.5)\Delta x\right)^2} \cdot \Delta x$$

## Implementation Highlights

* **256-bit SIMD Vectorization (`__m256d`):** Processes **4 double-precision floating-point values simultaneously** per iteration using Intel AVX2 intrinsics (`<immintrin.h>`).
* **Branchless Tail Handling (Masking):** Instead of relying on a separate scalar cleanup loop when `count` is not a multiple of 4, the loop generates a comparison mask (`_mm256_cmp_pd` with `_CMP_GT_OQ`) against the upper bound `end`. Out-of-bounds lanes are zeroed out via a bitwise AND (`_mm256_and_pd`) before accumulation.
* **Dual Timing Metrics:** Measures both **Process CPU time** (`std::clock`) and **Wall-clock time** (`std::chrono::steady_clock`) to provide an accurate comparison between scalar and vectorized execution.

## Prerequisites

* A C++11 compliant compiler (`g++` or `clang++`)
* An x86-64 CPU with **AVX2** instruction set support

## Compilation & Execution

Compile the source code with `-O3` optimizations and the `-mavx2` flag enabled:

```bash
g++ -std=c++11 -O3 -mavx2 pi_avx2.cpp -o pi_avx2
```

Run the executable:

```bash
./pi_avx2
```

*(Note: By default, `STEPS` is set to $2^{24}$ (`16,777,216` iterations). You can modify the `STEPS` constant in the source file to test different workloads or non-multiples of 4 to verify tail-masking accuracy.)*

