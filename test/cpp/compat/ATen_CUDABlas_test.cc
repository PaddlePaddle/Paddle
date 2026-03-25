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
#include "test/cpp/compat/cuda_test_utils.h"

// Helper: allocate three same-sized device buffers, copy host data in,
// invoke a kernel via |fn|, copy results back, synchronize, then free.
// |fn| receives (d_a, d_b, d_c); it must not free them.
template <typename T, typename Fn>
static void runOnDevice(const std::vector<T>& h_a,
                        const std::vector<T>& h_b,
                        std::vector<T>* h_c,
                        Fn fn) {
  size_t bytes = h_a.size() * sizeof(T);
  T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  ASSERT_EQ(cudaMalloc(&d_a, bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_b, bytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_c, bytes), cudaSuccess);

  ASSERT_EQ(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_c, h_c->data(), bytes, cudaMemcpyHostToDevice),
            cudaSuccess);

  fn(d_a, d_b, d_c);

  ASSERT_EQ(cudaMemcpy(h_c->data(), d_c, bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

// Runs 2x2 no-transpose gemm: C = alpha*A*B + beta*C and checks the result.
//
// Column-major layout:
//   A: col0={1,3}, col1={2,4}  =>  logical A = [[1,2],[3,4]]
//   B: col0={5,7}, col1={6,8}  =>  logical B = [[5,6],[7,8]]
//   A*B = [[19,22],[43,50]]    stored col-major: col0={19,43}, col1={22,50}
template <typename T, typename MathT = at::opmath_type<T>>
class GemmTester {
 public:
  static constexpr int64_t N = 2;

  static double toDouble(T val) { return static_cast<double>(val); }

  void Run() {
    SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
    std::vector<T> h_a = {T(1), T(3), T(2), T(4)};
    std::vector<T> h_b = {T(5), T(7), T(6), T(8)};
    std::vector<T> h_c(N * N, T(0));

    MathT alpha = static_cast<MathT>(1);
    MathT beta = static_cast<MathT>(0);

    runOnDevice(h_a, h_b, &h_c, [&](T* d_a, T* d_b, T* d_c) {
      at::cuda::blas::gemm<T>(
          'N', 'N', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);
    });

    EXPECT_NEAR(toDouble(h_c[0]), 19.0, 1e-2);  // C(0,0)
    EXPECT_NEAR(toDouble(h_c[1]), 43.0, 1e-2);  // C(1,0)
    EXPECT_NEAR(toDouble(h_c[2]), 22.0, 1e-2);  // C(0,1)
    EXPECT_NEAR(toDouble(h_c[3]), 50.0, 1e-2);  // C(1,1)
  }

  // transA='T': C = alpha * A^T * B + beta * C
  // A^T = [[1,3],[2,4]],  A^T * B = [[26,30],[38,44]]
  void RunTransA() {
    SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
    std::vector<T> h_a = {T(1), T(3), T(2), T(4)};
    std::vector<T> h_b = {T(5), T(7), T(6), T(8)};
    std::vector<T> h_c(N * N, T(0));

    MathT alpha = static_cast<MathT>(1);
    MathT beta = static_cast<MathT>(0);

    runOnDevice(h_a, h_b, &h_c, [&](T* d_a, T* d_b, T* d_c) {
      at::cuda::blas::gemm<T>(
          'T', 'N', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);
    });

    EXPECT_NEAR(toDouble(h_c[0]), 26.0, 1e-2);
    EXPECT_NEAR(toDouble(h_c[1]), 38.0, 1e-2);
    EXPECT_NEAR(toDouble(h_c[2]), 30.0, 1e-2);
    EXPECT_NEAR(toDouble(h_c[3]), 44.0, 1e-2);
  }
};

TEST(CUDABlasTest, GemmDouble) {
  GemmTester<double> t;
  t.Run();
}

TEST(CUDABlasTest, GemmDoubleTransA) {
  GemmTester<double> t;
  t.RunTransA();
}

TEST(CUDABlasTest, GemmFloat) {
  GemmTester<float> t;
  t.Run();
}

TEST(CUDABlasTest, GemmFloatTransA) {
  GemmTester<float> t;
  t.RunTransA();
}

TEST(CUDABlasTest, GemmFloatTransALowercase) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  constexpr int64_t N = 2;

  std::vector<float> h_a = {1.F, 3.F, 2.F, 4.F};
  std::vector<float> h_b = {5.F, 7.F, 6.F, 8.F};
  std::vector<float> h_c(N * N, 0.F);

  float alpha = 1.F;
  float beta = 0.F;
  runOnDevice(h_a, h_b, &h_c, [&](float* d_a, float* d_b, float* d_c) {
    at::cuda::blas::gemm<float>(
        't', 'n', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);
  });

  EXPECT_NEAR(h_c[0], 26.0f, 1e-3f);
  EXPECT_NEAR(h_c[1], 38.0f, 1e-3f);
  EXPECT_NEAR(h_c[2], 30.0f, 1e-3f);
  EXPECT_NEAR(h_c[3], 44.0f, 1e-3f);
}

TEST(CUDABlasTest, GemmComplexDouble) {
  GemmTester<c10::complex<double>> t;
  t.Run();
}

TEST(CUDABlasTest, GemmComplexFloat) {
  GemmTester<c10::complex<float>> t;
  t.Run();
}

TEST(CUDABlasTest, GemmHalf) {
  GemmTester<at::Half> t;
  t.Run();
}

TEST(CUDABlasTest, GemmBFloat16) {
  GemmTester<at::BFloat16> t;
  t.Run();
}

// to_cublas_op 'C'/'c' path: C = A^H * I = A^H (conjugate-transpose of A).
//
// A stored col-major: col0={1+i,2+2i}, col1={3+3i,4+4i}
// A^H stored col-major: col0={1-i,3-3i}, col1={2-2i,4-4i}
TEST(CUDABlasTest, GemmComplexFloatConjTrans) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  constexpr int64_t N = 2;
  using T = c10::complex<float>;

  std::vector<T> h_a = {T(1, 1), T(2, 2), T(3, 3), T(4, 4)};
  std::vector<T> h_b = {T(1, 0), T(0, 0), T(0, 0), T(1, 0)};  // identity
  std::vector<T> h_c(N * N, T(0, 0));

  float alpha = 1.0f;
  float beta = 0.0f;

  runOnDevice(h_a, h_b, &h_c, [&](T* d_a, T* d_b, T* d_c) {
    at::cuda::blas::gemm<T>(
        'C', 'N', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);
  });

  EXPECT_NEAR(h_c[0].real, 1.0f, 1e-3f);
  EXPECT_NEAR(h_c[0].imag, -1.0f, 1e-3f);
  EXPECT_NEAR(h_c[1].real, 3.0f, 1e-3f);
  EXPECT_NEAR(h_c[1].imag, -3.0f, 1e-3f);
  EXPECT_NEAR(h_c[2].real, 2.0f, 1e-3f);
  EXPECT_NEAR(h_c[2].imag, -2.0f, 1e-3f);
  EXPECT_NEAR(h_c[3].real, 4.0f, 1e-3f);
  EXPECT_NEAR(h_c[3].imag, -4.0f, 1e-3f);
}

// Same as above but uses lowercase 'c'/'n' to exercise that switch-case branch.
TEST(CUDABlasTest, GemmComplexDoubleConjTransLower) {
  SKIP_IF_CUDA_RUNTIME_UNAVAILABLE();
  constexpr int64_t N = 2;
  using T = c10::complex<double>;

  std::vector<T> h_a = {T(1, 1), T(2, 2), T(3, 3), T(4, 4)};
  std::vector<T> h_b = {T(1, 0), T(0, 0), T(0, 0), T(1, 0)};
  std::vector<T> h_c(N * N, T(0, 0));

  double alpha = 1.0;
  double beta = 0.0;

  runOnDevice(h_a, h_b, &h_c, [&](T* d_a, T* d_b, T* d_c) {
    at::cuda::blas::gemm<T>(
        'c', 'n', N, N, N, alpha, d_a, N, d_b, N, beta, d_c, N);
  });

  EXPECT_NEAR(h_c[0].real, 1.0, 1e-6);
  EXPECT_NEAR(h_c[0].imag, -1.0, 1e-6);
  EXPECT_NEAR(h_c[1].real, 3.0, 1e-6);
  EXPECT_NEAR(h_c[1].imag, -3.0, 1e-6);
}

TEST(CUDABlasTest, GemmInvalidTransposeThrows) {
  constexpr int64_t N = 1;
  double alpha = 1.0;
  double beta = 0.0;
  EXPECT_THROW(at::cuda::blas::gemm<double>('X',
                                            'N',
                                            N,
                                            N,
                                            N,
                                            alpha,
                                            static_cast<const double*>(nullptr),
                                            N,
                                            static_cast<const double*>(nullptr),
                                            N,
                                            beta,
                                            static_cast<double*>(nullptr),
                                            N),
               std::exception);
}

#endif  // PADDLE_WITH_CUDA
