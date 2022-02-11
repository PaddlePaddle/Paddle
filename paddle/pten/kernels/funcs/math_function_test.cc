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
#include "paddle/pten/kernels/funcs/math_function.h"
#include "gtest/gtest.h"
#include "paddle/fluid/operators/math/blas.h"

template <typename T>
inline paddle::operators::math::BlasT<paddle::platform::CPUDeviceContext, T>
GetBlas(const paddle::platform::CPUDeviceContext& context) {
  return paddle::operators::math::GetBlas<paddle::platform::CPUDeviceContext,
                                          T>(context);
}

TEST(math_function, gemm_notrans_cblas) {
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;

  int m = 2;
  int n = 3;
  int k = 3;
  auto* cpu_place = new paddle::platform::CPUPlace();
  float* input1_ptr = input1.mutable_data<float>({2, 3}, *cpu_place);
  float arr1[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr1, 6 * sizeof(float));
  float* input2_ptr = input2.mutable_data<float>({3, 4}, *cpu_place);
  float arr2[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  memcpy(input2_ptr, arr2, 12 * sizeof(float));
  float* input3_ptr = input3.mutable_data<float>({2, 4}, *cpu_place);
  float arr3[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  memcpy(input3_ptr, arr3, 8 * sizeof(float));

  paddle::platform::CPUDeviceContext context(*cpu_place);
  GetBlas<float>(context).GEMM(false,
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
  paddle::framework::Tensor mat_a;
  paddle::framework::Tensor mat_b;
  paddle::framework::Tensor mat_c_smm;
  paddle::framework::Tensor mat_c_mkl;
  auto* cpu_place = new paddle::platform::CPUPlace();

  T* A = mat_a.mutable_data<T>({m, k}, *cpu_place);
  T* B = mat_b.mutable_data<T>({k, n}, *cpu_place);
  T* CSMM = mat_c_smm.mutable_data<T>({m, n}, *cpu_place);
  T* CMKL = mat_c_mkl.mutable_data<T>({m, n}, *cpu_place);
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
    paddle::operators::math::CBlas<T>::SMM_GEMM(&transa,
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
    paddle::operators::math::CBlas<T>::GEMM(CblasRowMajor,
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
  paddle::framework::Tensor input1;
  paddle::framework::Tensor input2;
  paddle::framework::Tensor input3;

  int m = 2;
  int n = 3;
  int k = 3;
  auto* cpu_place = new paddle::platform::CPUPlace();
  float* input1_ptr = input1.mutable_data<float>({2, 3}, *cpu_place);
  float arr1[6] = {0, 1, 2, 3, 4, 5};
  memcpy(input1_ptr, arr1, 6 * sizeof(float));
  float* input2_ptr = input2.mutable_data<float>({4, 3}, *cpu_place);
  float arr2[12] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
  memcpy(input2_ptr, arr2, 12 * sizeof(float));
  float* input3_ptr = input3.mutable_data<float>({2, 4}, *cpu_place);
  float arr3[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  memcpy(input3_ptr, arr3, 8 * sizeof(float));

  paddle::platform::CPUDeviceContext context(*cpu_place);
  GetBlas<float>(context).GEMM(false,
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
  delete cpu_place;
  cpu_place = NULL;

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
  paddle::framework::Tensor tensor;
  auto* cpu_place = new paddle::platform::CPUPlace();
  float* t = tensor.mutable_data<float>({2, 2}, *cpu_place);
  paddle::platform::CPUDeviceContext context(*cpu_place);
  pten::funcs::SetConstant<paddle::platform::CPUDeviceContext, float> functor;
  functor(context, &tensor, 0);
  EXPECT_EQ(t[0], 0);
  EXPECT_EQ(t[1], 0);
  EXPECT_EQ(t[2], 0);
  EXPECT_EQ(t[3], 0);

  functor(context, &tensor, 1);

  EXPECT_EQ(t[0], 1);
  EXPECT_EQ(t[1], 1);
  EXPECT_EQ(t[2], 1);
  EXPECT_EQ(t[3], 1);
}

template <typename T>
void GemvTest(int m, int n, bool trans) {
  paddle::framework::Tensor mat_a;
  paddle::framework::Tensor vec_b;
  paddle::framework::Tensor vec_c;
  auto* cpu_place = new paddle::platform::CPUPlace();
  int b_num = trans ? m : n;
  int c_num = trans ? n : m;

  T* data_a = mat_a.mutable_data<T>({m, n}, *cpu_place);
  T* data_b = vec_b.mutable_data<T>({b_num}, *cpu_place);
  T* data_c = vec_c.mutable_data<T>({c_num}, *cpu_place);
  for (int i = 0; i < mat_a.numel(); ++i) {
    data_a[i] = static_cast<T>(i);
  }
  for (int i = 0; i < vec_b.numel(); ++i) {
    data_b[i] = static_cast<T>(i);
  }

  paddle::platform::CPUDeviceContext context(*cpu_place);
  GetBlas<T>(context).GEMV(trans,
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
  delete cpu_place;
}

TEST(math_function, gemv) {
  GemvTest<float>(3, 13, false);
  GemvTest<double>(4, 5, false);
  GemvTest<float>(12, 7, true);
  GemvTest<double>(7, 9, true);
}

TEST(math_funciton, set_constant) {
  paddle::framework::Tensor t;
  t.Resize({10, 10});
  t.mutable_data<int>(paddle::platform::CPUPlace());
  auto* ctx = new paddle::platform::CPUDeviceContext();
  ctx->Init();
  pten::funcs::set_constant(*ctx, &t, 10);
  for (int64_t i = 0; i < t.numel(); ++i) {
    PADDLE_ENFORCE_EQ(10,
                      t.data<int>()[i],
                      paddle::platform::errors::InvalidArgument(
                          "Each value of input tensor should be 10, "
                          "but received %d.",
                          t.data<int>()[i]));
  }
  delete ctx;
}

template <typename T>
void GemmWarpTest(int m, int n, int k, T alpha, T beta) {
  paddle::framework::Tensor mat_a;
  paddle::framework::Tensor mat_b;
  paddle::framework::Tensor mat_c_ref;
  paddle::framework::Tensor mat_c_mkl;
  auto* cpu_place = new paddle::platform::CPUPlace();

  T* A = mat_a.mutable_data<T>({m, k}, *cpu_place);
  T* B = mat_b.mutable_data<T>({k, n}, *cpu_place);
  T* CREF = mat_c_ref.mutable_data<T>({m, n}, *cpu_place);
  T* CMKL = mat_c_mkl.mutable_data<T>({m, n}, *cpu_place);

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
  paddle::platform::CPUDeviceContext context(*cpu_place);
  GetBlas<T>(context).GEMM(
      CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, B, beta, CREF);

  // lda,ldb,ldc follow RowMajor
  int lda = k;
  int ldb = n;
  int ldc = n;
  paddle::operators::math::CBlas<T>::GEMM(CblasRowMajor,
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
  delete cpu_place;
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
