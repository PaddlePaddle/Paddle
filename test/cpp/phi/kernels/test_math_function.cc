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
#include <cstdint>
#include <limits>
#include <set>

#include "gtest/gtest.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"
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

TEST(math_function, blas_int64_dimensions) {
  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());
  auto blas = GetBlas<float>(*dev_ctx);

  const int64_t m = 2;
  const int64_t n = 2;
  const int64_t k = 3;
  const std::array<float, 6> a = {1, 2, 3, 4, 5, 6};
  const std::array<float, 6> b = {1, 2, 3, 4, 5, 6};
  const std::array<float, 4> expected = {22, 28, 49, 64};

  std::array<float, 4> matmul_out{};
  blas.MatMul(m, n, k, a.data(), b.data(), matmul_out.data());
  EXPECT_EQ(matmul_out, expected);

  const std::array<float, 3> vector = {1, 1, 1};
  const std::array<float, 2> expected_gemv = {6, 15};
  std::array<float, 2> gemv_out{};
  blas.GEMV(false, m, k, 1.0f, a.data(), vector.data(), 0.0f, gemv_out.data());
  EXPECT_EQ(gemv_out, expected_gemv);

  std::array<float, 4> bool_gemm_out{};
  blas.GEMM(false,
            false,
            m,
            n,
            k,
            1.0f,
            a.data(),
            k,
            b.data(),
            n,
            0.0f,
            bool_gemm_out.data(),
            n);
  EXPECT_EQ(bool_gemm_out, expected);

  std::array<float, 4> enum_gemm_out{};
  blas.GEMM(CblasNoTrans,
            CblasNoTrans,
            m,
            n,
            k,
            1.0f,
            a.data(),
            k,
            b.data(),
            n,
            0.0f,
            enum_gemm_out.data(),
            n);
  EXPECT_EQ(enum_gemm_out, expected);

  const std::array<float, 2> batch_a0 = {2, 3};
  const std::array<float, 2> batch_a1 = {4, 5};
  const std::array<float, 2> batch_b0 = {6, 7};
  const std::array<float, 2> batch_b1 = {8, 9};
  std::array<float, 1> batch_c0{};
  std::array<float, 1> batch_c1{};
  const float* batch_a[] = {batch_a0.data(), batch_a1.data()};
  const float* batch_b[] = {batch_b0.data(), batch_b1.data()};
  float* batch_c[] = {batch_c0.data(), batch_c1.data()};
  const int64_t batch_count = 2;
  blas.BatchedGEMM(CblasNoTrans,
                   CblasNoTrans,
                   1,
                   1,
                   2,
                   1.0f,
                   batch_a,
                   batch_b,
                   0.0f,
                   batch_c,
                   batch_count);
  EXPECT_FLOAT_EQ(batch_c0[0], 33.0f);
  EXPECT_FLOAT_EQ(batch_c1[0], 77.0f);
}

TEST(math_function, blas_rejects_unsupported_large_dimensions) {
  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());
  auto blas = GetBlas<float>(*dev_ctx);
  float value = 1.0f;
  const int64_t too_large =
      static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;

  EXPECT_THROW(blas.MatMul(too_large, 1, 1, &value, &value, &value),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(
      blas.GEMV(false, too_large, 1, 1.0f, &value, &value, 0.0f, &value),
      common::enforce::EnforceNotMet);
  EXPECT_THROW(blas.GEMM(CblasNoTrans,
                         CblasNoTrans,
                         -1,
                         1,
                         1,
                         1.0f,
                         &value,
                         &value,
                         0.0f,
                         &value),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(blas.GEMM(false,
                         false,
                         1,
                         1,
                         1,
                         1.0f,
                         &value,
                         too_large,
                         &value,
                         1,
                         0.0f,
                         &value,
                         1),
               common::enforce::EnforceNotMet);

  const float** null_inputs = nullptr;
  float** null_outputs = nullptr;
  EXPECT_THROW(blas.BatchedGEMM(CblasNoTrans,
                                CblasNoTrans,
                                1,
                                1,
                                1,
                                1.0f,
                                null_inputs,
                                null_inputs,
                                0.0f,
                                null_outputs,
                                too_large),
               common::enforce::EnforceNotMet);
#ifdef PADDLE_WITH_MKLML
  EXPECT_THROW(blas.GEMM_ALLOC(CblasBMatrix, too_large, 1, 1),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(
      blas.GEMM_PACK(
          CblasBMatrix, CblasNoTrans, 1, 1, 1, 1.0f, &value, too_large, &value),
      common::enforce::EnforceNotMet);
  EXPECT_THROW(blas.GEMM_COMPUTE(CblasNoTrans,
                                 CblasPacked,
                                 1,
                                 1,
                                 1,
                                 &value,
                                 too_large,
                                 &value,
                                 1,
                                 0.0f,
                                 &value,
                                 1),
               common::enforce::EnforceNotMet);
#endif
#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
  phi::funcs::Blas<phi::CPUContext> base_blas(*dev_ctx);
  EXPECT_THROW(base_blas.BatchedGEMMWithHead<float>(CblasNoTrans,
                                                    CblasNoTrans,
                                                    too_large,
                                                    1,
                                                    1,
                                                    1,
                                                    1.0f,
                                                    &value,
                                                    &value,
                                                    0.0f,
                                                    &value,
                                                    1,
                                                    1,
                                                    1,
                                                    1,
                                                    false),
               common::enforce::EnforceNotMet);

  phi::DenseTensor empty;
  const phi::funcs::MatDescriptor unit_matrix{1, 1, 0, 0, false};
  const int64_t invalid_head_number = 0;
  EXPECT_THROW(base_blas.MatMulWithHead<float>(empty,
                                               unit_matrix,
                                               empty,
                                               unit_matrix,
                                               1.0f,
                                               invalid_head_number,
                                               &empty,
                                               0.0f,
                                               false),
               common::enforce::EnforceNotMet);
#endif
}

TEST(math_function, blas_dimension_validation_is_overflow_safe) {
  const int64_t int64_max = std::numeric_limits<int64_t>::max();

  EXPECT_EQ(phi::funcs::detail::checked_blas_mul(int64_max, 1, "test"),
            int64_max);
  EXPECT_THROW(phi::funcs::detail::checked_blas_mul(int64_max, 2, "test"),
               common::enforce::EnforceNotMet);
  EXPECT_THROW(phi::funcs::detail::to_blas_int(-1, "test"),
               common::enforce::EnforceNotMet);
}

TEST(math_function, fc_functor_keeps_legacy_and_int64_signatures) {
  using Functor = phi::funcs::FCFunctor<phi::CPUContext, float>;
  using LegacyOperator = void (Functor::*)(const phi::CPUContext&,
                                           int,
                                           int,
                                           int,
                                           const float*,
                                           const float*,
                                           float*,
                                           const float*,
                                           bool,
                                           bool);
  using Int64Operator = void (Functor::*)(const phi::CPUContext&,
                                          int64_t,
                                          int64_t,
                                          int64_t,
                                          const float*,
                                          const float*,
                                          float*,
                                          const float*,
                                          bool,
                                          bool);

  auto legacy_operator = static_cast<LegacyOperator>(&Functor::operator());
  auto int64_operator = static_cast<Int64Operator>(&Functor::operator());
  EXPECT_TRUE(legacy_operator != nullptr);
  EXPECT_TRUE(int64_operator != nullptr);

  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());
  float value = 1.0f;
  const int64_t invalid_padded_width =
      static_cast<int64_t>(std::numeric_limits<int>::max()) - 3;
  Functor fc;
  EXPECT_THROW(fc(*dev_ctx,
                  int64_t{1},
                  invalid_padded_width,
                  int64_t{1},
                  &value,
                  &value,
                  &value,
                  nullptr,
                  false,
                  true),
               common::enforce::EnforceNotMet);
}

TEST(math_function, dot_with_blas_zero_length) {
  bool called = false;
  auto result = phi::funcs::detail::dot_with_blas<float>(
      0,
      nullptr,
      1,
      nullptr,
      1,
      [&](int, const float*, int, const float*, int) {
        called = true;
        return 1.0f;
      });

  EXPECT_FALSE(called);
  EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(math_function, dot_with_blas_single_element) {
  const std::array<float, 1> x = {3.0f};
  const std::array<float, 1> y = {4.0f};
  bool called = false;

  auto result = phi::funcs::detail::dot_with_blas<float>(
      1,
      x.data(),
      8,
      y.data(),
      9,
      [&](int n, const float* px, int incx, const float* py, int incy) {
        called = true;
        EXPECT_EQ(n, 1);
        EXPECT_EQ(incx, 1);
        EXPECT_EQ(incy, 1);
        return px[0] * py[0];
      });

  EXPECT_TRUE(called);
  EXPECT_FLOAT_EQ(result, 12.0f);
}

TEST(math_function, dot_fallback_strided) {
  const std::array<float, 5> x = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f};
  const std::array<float, 7> y = {4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 6.0f};

  auto result =
      phi::funcs::detail::dot_fallback<float>(3, x.data(), 2, y.data(), 3);

  EXPECT_FLOAT_EQ(result, 32.0f);
}

TEST(math_function, level1_blas_compatible_true) {
  EXPECT_TRUE(phi::funcs::detail::level1_blas_compatible(0, 1, 1));
  EXPECT_TRUE(phi::funcs::detail::level1_blas_compatible(
      std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0));
}

TEST(math_function, level1_blas_compatible_false) {
  EXPECT_FALSE(phi::funcs::detail::level1_blas_compatible(-1, 1, 1));
  EXPECT_FALSE(phi::funcs::detail::level1_blas_compatible(
      static_cast<int64_t>(std::numeric_limits<int>::max()) + 1, 1, 1));
  EXPECT_FALSE(phi::funcs::detail::level1_blas_compatible(
      1, static_cast<int64_t>(std::numeric_limits<int>::min()) - 1, 1));
  EXPECT_FALSE(phi::funcs::detail::level1_blas_compatible(
      1, 1, static_cast<int64_t>(std::numeric_limits<int>::max()) + 1));
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
void GemvTest(int64_t m, int64_t n, bool trans) {
  phi::DenseTensor mat_a;
  phi::DenseTensor vec_b;
  phi::DenseTensor vec_c;
  int64_t b_num = trans ? m : n;
  int64_t c_num = trans ? n : m;

  auto* dev_ctx =
      phi::DeviceContextPool::Instance().GetByPlace(phi::CPUPlace());

  mat_a.Resize({m, n});
  T* data_a = dev_ctx->template Alloc<T>(&mat_a);
  vec_b.Resize({b_num});
  T* data_b = dev_ctx->template Alloc<T>(&vec_b);
  vec_c.Resize({c_num});
  T* data_c = dev_ctx->template Alloc<T>(&vec_c);
  for (int64_t i = 0; i < mat_a.numel(); ++i) {
    data_a[i] = static_cast<T>(i);
  }
  for (int64_t i = 0; i < vec_b.numel(); ++i) {
    data_b[i] = static_cast<T>(i);
  }

  GetBlas<T>(*dev_ctx).GEMV(trans, m, n, 1., data_a, data_b, 0., data_c);

  if (!trans) {
    for (int64_t i = 0; i < m; ++i) {
      T sum = 0.0;
      for (int64_t j = 0; j < n; ++j) {
        sum += data_a[i * n + j] * data_b[j];
      }
      ASSERT_FLOAT_EQ(data_c[i], sum);
    }
  } else {
    for (int64_t i = 0; i < n; ++i) {
      T sum = 0.0;
      for (int64_t j = 0; j < m; ++j) {
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
