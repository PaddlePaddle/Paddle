// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(PADDLE_WITH_CUDA)

#include <ATen/cuda/CUDABlas.h>

#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"

// ---------------------------------------------------------------------------
// Helper: run a simple 2×2 gemm  C = alpha * A * B + beta * C  (no‐trans)
// and verify the result on the host side.
//
// A = | 1  2 |   B = | 5  6 |   =>  A*B = | 19 22 |
//     | 3  4 |       | 7  8 |              | 43 50 |
//
// With alpha = 1, beta = 0 we expect C == A*B.
// ---------------------------------------------------------------------------

template <typename T, typename MathT = at::opmath_type<T>>
class GemmTester {
 public:
  static constexpr int64_t N = 2;

  void Run() {
    // Host matrices (column‐major for BLAS)
    // Column‐major layout: col0 = {1,3}, col1 = {2,4}
    std::vector<T> h_a = {T(1), T(3), T(2), T(4)};
    std::vector<T> h_b = {T(5), T(7), T(6), T(8)};
    std::vector<T> h_c(N * N, T(0));

    T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = N * N * sizeof(T);

    ASSERT_EQ(cudaMalloc(&d_a, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice),
              cudaSuccess);

    MathT alpha = static_cast<MathT>(1);
    MathT beta = static_cast<MathT>(0);

    // Call gemm: 'N', 'N' → no transpose on either matrix
    at::cuda::blas::gemm<T>(
        'N', 'N', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);

    ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Expected C (column‐major): col0={19,43}, col1={22,50}
    auto approx = [](T val) -> double { return static_cast<double>(val); };

    EXPECT_NEAR(approx(h_c[0]), 19.0, 1e-2);  // C(0,0)
    EXPECT_NEAR(approx(h_c[1]), 43.0, 1e-2);  // C(1,0)
    EXPECT_NEAR(approx(h_c[2]), 22.0, 1e-2);  // C(0,1)
    EXPECT_NEAR(approx(h_c[3]), 50.0, 1e-2);  // C(1,1)

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }

  // Test with transA = 'T' to also cover the 'T' path in to_cublas_op
  void RunTransA() {
    // A stored col‐major: col0={1,3}, col1={2,4}  → logical A = [[1,2],[3,4]]
    // BLAS uses op(A)=A^T = [[1,3],[2,4]]
    std::vector<T> h_a = {T(1), T(3), T(2), T(4)};
    std::vector<T> h_b = {T(5), T(7), T(6), T(8)};
    std::vector<T> h_c(N * N, T(0));

    T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = N * N * sizeof(T);

    ASSERT_EQ(cudaMalloc(&d_a, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c, bytes), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice),
              cudaSuccess);

    MathT alpha = static_cast<MathT>(1);
    MathT beta = static_cast<MathT>(0);

    // transA='T', transB='N'  →  C = alpha * A^T * B + beta * C
    at::cuda::blas::gemm<T>(
        'T', 'N', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);

    ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // A^T = | 1  3 |   B = | 5  6 |
    //        | 2  4 |       | 7  8 |
    // A^T * B = | 1*5+3*7  1*6+3*8 | = | 26 30 |
    //           | 2*5+4*7  2*6+4*8 |   | 38 44 |
    auto approx = [](T val) -> double { return static_cast<double>(val); };
    EXPECT_NEAR(approx(h_c[0]), 26.0, 1e-2);
    EXPECT_NEAR(approx(h_c[1]), 38.0, 1e-2);
    EXPECT_NEAR(approx(h_c[2]), 30.0, 1e-2);
    EXPECT_NEAR(approx(h_c[3]), 44.0, 1e-2);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  }
};

// ---------------------------------------------------------------------------
// gemm<double>
// ---------------------------------------------------------------------------
TEST(CUDABlasTest, GemmDouble) {
  GemmTester<double> t;
  t.Run();
}

TEST(CUDABlasTest, GemmDoubleTransA) {
  GemmTester<double> t;
  t.RunTransA();
}

// ---------------------------------------------------------------------------
// gemm<float>
// ---------------------------------------------------------------------------
TEST(CUDABlasTest, GemmFloat) {
  GemmTester<float> t;
  t.Run();
}

TEST(CUDABlasTest, GemmFloatTransA) {
  GemmTester<float> t;
  t.RunTransA();
}

// ---------------------------------------------------------------------------
// gemm<c10::complex<double>>
// ---------------------------------------------------------------------------
TEST(CUDABlasTest, GemmComplexDouble) {
  GemmTester<c10::complex<double>> t;
  t.Run();
}

// ---------------------------------------------------------------------------
// gemm<c10::complex<float>>
// ---------------------------------------------------------------------------
TEST(CUDABlasTest, GemmComplexFloat) {
  GemmTester<c10::complex<float>> t;
  t.Run();
}

// ---------------------------------------------------------------------------
// gemm<at::Half>
// ---------------------------------------------------------------------------
TEST(CUDABlasTest, GemmHalf) {
  GemmTester<at::Half> t;
  t.Run();
}

// ---------------------------------------------------------------------------
// gemm<at::BFloat16>
// ---------------------------------------------------------------------------
TEST(CUDABlasTest, GemmBFloat16) {
  GemmTester<at::BFloat16> t;
  t.Run();
}

// ---------------------------------------------------------------------------
// to_cublas_op — exercise all valid characters (both cases) via gemm calls.
// The 'N'/'n' and 'T'/'t' paths are covered above. Cover 'c'/'C' via the
// complex<float> specialisation (conjugate‐transpose for Hermitian use).
// ---------------------------------------------------------------------------
TEST(CUDABlasTest, GemmComplexFloatConjTrans) {
  constexpr int64_t N = 2;
  using T = c10::complex<float>;

  // A stored col‐major; we pass transA='C' so BLAS uses A^H.
  std::vector<T> h_a = {T(1, 1), T(2, 2), T(3, 3), T(4, 4)};
  std::vector<T> h_b = {T(1, 0), T(0, 0), T(0, 0), T(1, 0)};  // identity
  std::vector<T> h_c(N * N, T(0, 0));

  T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  size_t bytes = N * N * sizeof(T);

  ASSERT_EQ(cudaMalloc(&d_a, bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_b, bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_c, bytes), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);

  float alpha = 1.0f;
  float beta = 0.0f;

  // C = A^H * I = conjugate‐transpose of A
  at::cuda::blas::gemm<T>(
      'C', 'N', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);

  ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // A stored col‐major: col0={1+i,2+2i}, col1={3+3i,4+4i}
  // Logical A:  row0=(1+i, 3+3i)   row1=(2+2i, 4+4i)
  // A^H: conjugate‐transpose
  //   row0 = conj(col0 of A) = (1-i, 2-2i)
  //   row1 = conj(col1 of A) = (3-3i, 4-4i)
  // A^H * I = A^H, stored col‐major: col0={1-i, 3-3i}, col1={2-2i, 4-4i}
  auto real = [](T v) -> float { return v.real; };
  auto imag = [](T v) -> float { return v.imag; };

  EXPECT_NEAR(real(h_c[0]), 1.0f, 1e-3f);
  EXPECT_NEAR(imag(h_c[0]), -1.0f, 1e-3f);
  EXPECT_NEAR(real(h_c[1]), 3.0f, 1e-3f);
  EXPECT_NEAR(imag(h_c[1]), -3.0f, 1e-3f);
  EXPECT_NEAR(real(h_c[2]), 2.0f, 1e-3f);
  EXPECT_NEAR(imag(h_c[2]), -2.0f, 1e-3f);
  EXPECT_NEAR(real(h_c[3]), 4.0f, 1e-3f);
  EXPECT_NEAR(imag(h_c[3]), -4.0f, 1e-3f);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

// ---------------------------------------------------------------------------
// to_cublas_op — lowercase 'c' path  (same logic, just exercises the
// switch‐case branch with lowercase letter)
// ---------------------------------------------------------------------------
TEST(CUDABlasTest, GemmComplexDoubleConjTransLower) {
  constexpr int64_t N = 2;
  using T = c10::complex<double>;

  std::vector<T> h_a = {T(1, 1), T(2, 2), T(3, 3), T(4, 4)};
  std::vector<T> h_b = {T(1, 0), T(0, 0), T(0, 0), T(1, 0)};
  std::vector<T> h_c(N * N, T(0, 0));

  T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  size_t bytes = N * N * sizeof(T);

  ASSERT_EQ(cudaMalloc(&d_a, bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_b, bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_c, bytes), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);

  double alpha = 1.0;
  double beta = 0.0;

  at::cuda::blas::gemm<T>(
      'c', 'n', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);

  ASSERT_EQ(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto real = [](T v) -> double { return v.real; };
  auto imag = [](T v) -> double { return v.imag; };

  EXPECT_NEAR(real(h_c[0]), 1.0, 1e-6);
  EXPECT_NEAR(imag(h_c[0]), -1.0, 1e-6);
  EXPECT_NEAR(real(h_c[1]), 3.0, 1e-6);
  EXPECT_NEAR(imag(h_c[1]), -3.0, 1e-6);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

#endif  // PADDLE_WITH_CUDA
