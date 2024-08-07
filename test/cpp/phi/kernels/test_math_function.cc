//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <set>

#include "gtest/gtest.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace tests {

template <typename T>
inline phi::funcs::BlasT<phi::CPUContext, T> GetBlas(
    const phi::CPUContext& context) {
  return phi::funcs::GetBlas<phi::CPUContext, T>(context);
}

TEST(math_function, gemm_notrans_cblas) {
  phi::DenseTensor input1;
  phi::DenseTensor input2;
  phi::DenseTensor input3;

  int m = 2;
  int n = 3;
  int k = 3;
  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());

  input1.Resize({2, 3});
  float* input1_ptr = dev_ctx->template Alloc<float>(&input1);
  std::array<float, 6> arr1 = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr1.data(), 6 * sizeof(float));
  input2.Resize({3, 4});
  float* input2_ptr = dev_ctx->template Alloc<float>(&input2);
  std::array<float, 12> arr2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  memcpy(input2_ptr, arr2.data(), 12 * sizeof(float));
  input3.Resize({2, 4});
  float* input3_ptr = dev_ctx->template Alloc<float>(&input3);
  std::array<float, 8> arr3 = {0, 1, 2, 3, 4, 5, 6, 7};
  memcpy(input3_ptr, arr3.data(), 8 * sizeof(float));

  GetBlas<float>(*dev_ctx).GEMM(false,
                                false,
                                m,
                                n,
                                k,
                                1,
                                input1_ptr,
                                3,
                                input2_ptr + 1,
                                4,
                                1,
                                input3_ptr + 1,
                                4);

  EXPECT_EQ(input3_ptr[0], 0);
  EXPECT_EQ(input3_ptr[1], 24);
  EXPECT_EQ(input3_ptr[2], 28);
  EXPECT_EQ(input3_ptr[3], 32);
  EXPECT_EQ(input3_ptr[4], 4);
  EXPECT_EQ(input3_ptr[5], 73);
  EXPECT_EQ(input3_ptr[6], 86);
  EXPECT_EQ(input3_ptr[7], 99);
}
#ifdef PADDLE_WITH_LIBXSMM
template <typename T>
void MklSmmCompare(int m, int n, int k) {
  phi::DenseTensor mat_a;
  phi::DenseTensor mat_b;
  phi::DenseTensor mat_c_smm;
  phi::DenseTensor mat_c_mkl;

  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());

  mat_a.Resize({m, k});
  T* A = dev_ctx->template Alloc<T>(&mat_a);
  mat_b.Resize({k, n});
  T* B = dev_ctx->template Alloc<T>(&mat_b);
  mat_c_smm.Resize({m, n});
  T* CSMM = dev_ctx->template Alloc<T>(&mat_c_smm);
  mat_c_mkl.Resize({m, n});
  T* CMKL = dev_ctx->template Alloc<T>(&mat_c_mkl);
  T alpha = static_cast<T>(1);
  T beta = static_cast<T>(0);
  for (int i = 0; i < mat_a.numel(); ++i) {
    A[i] = static_cast<T>(i);
  }
  for (int i = 0; i < mat_b.numel(); ++i) {
    B[i] = static_cast<T>(i);
  }
  // lda,ldb,ldc follow RowMajor
  int lda = k;
  int ldb = n;
  int ldc = n;

  auto smm = [&, m, n, k, lda, ldb, ldc, alpha, beta]() {
    const char transa = 'N';
    const char transb = 'N';
    phi::funcs::CBlas<T>::SMM_GEMM(&transa,
                                   &transb,
                                   &n,
                                   &m,
                                   &k,
                                   &alpha,
                                   B,
                                   &ldb,
                                   A,
                                   &lda,
                                   &beta,
                                   CSMM,
                                   &ldc);
  };

  auto mkl = [&, m, n, k, lda, ldb, ldc, alpha, beta]() {
    phi::funcs::CBlas<T>::GEMM(CblasRowMajor,
                               CblasNoTrans,
                               CblasNoTrans,
                               m,
                               n,
                               k,
                               alpha,
                               A,
                               lda,
                               B,
                               ldb,
                               beta,
                               CMKL,
                               ldc);
  };

  smm();
  mkl();
  ASSERT_EQ(mat_c_mkl.numel(), mat_c_smm.numel());
  for (int i = 0; i < mat_c_mkl.numel(); ++i) {
    EXPECT_FLOAT_EQ(CSMM[i], CMKL[i]);
  }
}
TEST(math_function, gemm_mkl_vs_smm) {
  MklSmmCompare<float>(1, 2, 3);
  MklSmmCompare<double>(1, 2, 3);
  MklSmmCompare<float>(3, 2, 1);
  MklSmmCompare<double>(3, 2, 1);
  MklSmmCompare<float>(3, 8, 5);
  MklSmmCompare<double>(3, 8, 5);
}
#endif

TEST(math_function, gemm_trans_cblas) {
  phi::DenseTensor input1;
  phi::DenseTensor input2;
  phi::DenseTensor input3;

  int m = 2;
  int n = 3;
  int k = 3;
  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());

  input1.Resize({2, 3});
  float* input1_ptr = dev_ctx->template Alloc<float>(&input1);
  std::array<float, 6> arr1 = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr1.data(), 6 * sizeof(float));
  input2.Resize({4, 3});
  float* input2_ptr = dev_ctx->template Alloc<float>(&input2);
  std::array<float, 12> arr2 = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
  memcpy(input2_ptr, arr2.data(), 12 * sizeof(float));
  input3.Resize({2, 4});
  float* input3_ptr = dev_ctx->template Alloc<float>(&input3);
  std::array<float, 8> arr3 = {0, 1, 2, 3, 4, 5, 6, 7};
  memcpy(input3_ptr, arr3.data(), 8 * sizeof(float));

  GetBlas<float>(*dev_ctx).GEMM(false,
                                true,
                                m,
                                n,
                                k,
                                1,
                                input1_ptr,
                                3,
                                input2_ptr + 3,
                                3,
                                1,
                                input3_ptr + 1,
                                4);

  EXPECT_EQ(input3_ptr[0], 0);
  EXPECT_EQ(input3_ptr[1], 24);
  EXPECT_EQ(input3_ptr[2], 28);
  EXPECT_EQ(input3_ptr[3], 32);
  EXPECT_EQ(input3_ptr[4], 4);
  EXPECT_EQ(input3_ptr[5], 73);
  EXPECT_EQ(input3_ptr[6], 86);
  EXPECT_EQ(input3_ptr[7], 99);
}

TEST(math_function, zero) {
  phi::DenseTensor tensor;
  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());

  tensor.Resize({2, 2});
  float* t = dev_ctx->template Alloc<float>(&tensor);
  phi::funcs::SetConstant<phi::CPUContext, float> functor;
  functor(*dev_ctx, &tensor, 0);
  EXPECT_EQ(t[0], 0);
  EXPECT_EQ(t[1], 0);
  EXPECT_EQ(t[2], 0);
  EXPECT_EQ(t[3], 0);

  functor(*dev_ctx, &tensor, 1);

  EXPECT_EQ(t[0], 1);
  EXPECT_EQ(t[1], 1);
  EXPECT_EQ(t[2], 1);
  EXPECT_EQ(t[3], 1);
}

template <typename T>
void GemvTest(int m, int n, bool trans) {
  phi::DenseTensor mat_a;
  phi::DenseTensor vec_b;
  phi::DenseTensor vec_c;
  int b_num = trans ? m : n;
  int c_num = trans ? n : m;

  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());

  mat_a.Resize({m, n});
  T* data_a = dev_ctx->template Alloc<T>(&mat_a);
  vec_b.Resize({b_num});
  T* data_b = dev_ctx->template Alloc<T>(&vec_b);
  vec_c.Resize({c_num});
  T* data_c = dev_ctx->template Alloc<T>(&vec_c);
  for (int i = 0; i < mat_a.numel(); ++i) {
    data_a[i] = static_cast<T>(i);
  }
  for (int i = 0; i < vec_b.numel(); ++i) {
    data_b[i] = static_cast<T>(i);
  }

  GetBlas<T>(*dev_ctx).GEMV(trans,
                            static_cast<int>(m),
                            static_cast<int>(n),
                            1.,
                            data_a,
                            data_b,
                            0.,
                            data_c);

  if (!trans) {
    for (int i = 0; i < m; ++i) {
      T sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += data_a[i * n + j] * data_b[j];
      }
      ASSERT_FLOAT_EQ(data_c[i], sum);
    }
  } else {
    for (int i = 0; i < n; ++i) {
      T sum = 0.0;
      for (int j = 0; j < m; ++j) {
        sum += data_a[j * n + i] * data_b[j];
      }
      ASSERT_FLOAT_EQ(data_c[i], sum);
    }
  }
}

TEST(math_function, gemv) {
  GemvTest<float>(3, 13, false);
  GemvTest<double>(4, 5, false);
  GemvTest<float>(12, 7, true);
  GemvTest<double>(7, 9, true);
}

TEST(math_function, set_constant) {
  phi::DenseTensor t;
  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());
  t.Resize({10, 10});
  dev_ctx->template Alloc<int>(&t);
  phi::funcs::set_constant(*dev_ctx, &t, static_cast<int>(10));
  for (int64_t i = 0; i < t.numel(); ++i) {
    PADDLE_ENFORCE_EQ(10,
                      t.data<int>()[i],
                      common::errors::InvalidArgument(
                          "Each value of input tensor should be 10, "
                          "but received %d.",
                          t.data<int>()[i]));
  }
}

template <typename T>
void GemmWarpTest(int m, int n, int k, T alpha, T beta) {
  phi::DenseTensor mat_a;
  phi::DenseTensor mat_b;
  phi::DenseTensor mat_c_ref;
  phi::DenseTensor mat_c_mkl;

  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());

  mat_a.Resize({m, k});
  T* A = dev_ctx->template Alloc<T>(&mat_a);
  mat_b.Resize({k, n});
  T* B = dev_ctx->template Alloc<T>(&mat_b);
  mat_c_ref.Resize({m, n});
  T* CREF = dev_ctx->template Alloc<T>(&mat_c_ref);
  mat_c_mkl.Resize({m, n});
  T* CMKL = dev_ctx->template Alloc<T>(&mat_c_mkl);

  ASSERT_EQ(mat_c_mkl.numel(), mat_c_ref.numel());
  for (int i = 0; i < mat_a.numel(); ++i) {
    A[i] = static_cast<T>(i);
  }
  for (int i = 0; i < mat_b.numel(); ++i) {
    B[i] = static_cast<T>(i + 1);
  }
  for (int i = 0; i < mat_c_ref.numel(); ++i) {
    CREF[i] = static_cast<T>(i + 2);
    CMKL[i] = CREF[i];
  }

  // this would call gemm_warp
  GetBlas<T>(*dev_ctx).GEMM(
      CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, B, beta, CREF);

  // lda,ldb,ldc follow RowMajor
  int lda = k;
  int ldb = n;
  int ldc = n;
  phi::funcs::CBlas<T>::GEMM(CblasRowMajor,
                             CblasNoTrans,
                             CblasNoTrans,
                             m,
                             n,
                             k,
                             alpha,
                             A,
                             lda,
                             B,
                             ldb,
                             beta,
                             CMKL,
                             ldc);

  for (int i = 0; i < mat_c_mkl.numel(); ++i) {
    EXPECT_FLOAT_EQ(CREF[i], CMKL[i]);
  }
}

TEST(math_function, gemm_warp) {
  GemmWarpTest<float>(3, 2, 5, 1.f, 0.f);
  GemmWarpTest<float>(3, 2, 5, 2.f, 1.f);
  GemmWarpTest<float>(8, 5, 6, 1.f, 0.f);
  GemmWarpTest<float>(8, 5, 6, 2.f, 1.f);
  GemmWarpTest<double>(3, 2, 5, 1.0, 0.0);
  GemmWarpTest<double>(3, 2, 5, 2.0, 1.0);
  GemmWarpTest<double>(8, 5, 6, 1.0, 0.0);
  GemmWarpTest<double>(8, 5, 6, 2.0, 1.0);
}

}  // namespace tests
}  // namespace phi
