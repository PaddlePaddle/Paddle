// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/runtime/cpu/cblas.h"

#include <vector>

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/common/enforce.h"

namespace {

inline CBLAS_TRANSPOSE ToCblasTranspose(bool trans) {
  return trans ? CblasTrans : CblasNoTrans;
}

}  // namespace

void cinn_cpu_mkl_gemm_fp32(float alpha,
                            int M,
                            int N,
                            int K,
                            bool ta,
                            bool tb,
                            int lda,
                            int ldb,
                            int ldc,
                            float beta,
                            cinn_buffer_t* A,
                            cinn_buffer_t* B,
                            cinn_buffer_t* C) {
  cblas_sgemm(CblasRowMajor,
              ToCblasTranspose(ta),
              ToCblasTranspose(tb),
              M,
              N,
              K,
              alpha,
              reinterpret_cast<float*>(A->memory),
              lda,
              reinterpret_cast<float*>(B->memory),
              ldb,
              beta,
              reinterpret_cast<float*>(C->memory),
              ldc);
}

void cinn_cpu_mkl_gemm_batch_fp32(float alpha,
                                  int batch_size,
                                  int M,
                                  int N,
                                  int K,
                                  bool ta,
                                  bool tb,
                                  int lda,
                                  int ldb,
                                  int ldc,
                                  int a_stride,
                                  int b_stride,
                                  int c_stride,
                                  float beta,
                                  cinn_buffer_t* A,
                                  cinn_buffer_t* B,
                                  cinn_buffer_t* C) {
  std::vector<const float*> A_array(batch_size);
  std::vector<const float*> B_array(batch_size);
  std::vector<float*> C_array(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    A_array[i] = reinterpret_cast<float*>(A->memory) + i * a_stride;
    B_array[i] = reinterpret_cast<float*>(B->memory) + i * b_stride;
    C_array[i] = reinterpret_cast<float*>(C->memory) + i * c_stride;
  }
  CBLAS_TRANSPOSE trans_a = ToCblasTranspose(ta);
  CBLAS_TRANSPOSE trans_b = ToCblasTranspose(tb);
  cblas_sgemm_batch(CblasRowMajor,
                    &trans_a,
                    &trans_b,
                    &M,
                    &N,
                    &K,
                    &alpha,
                    A_array.data(),
                    &lda,
                    B_array.data(),
                    &ldb,
                    &beta,
                    C_array.data(),
                    &ldc,
                    1,
                    &batch_size);
}

/**
 * This function is temporarily unavailable, see the error message in the
 * following PR for details. The specific reason may be that the custom call
 * does not support host op. See: https://github.com/PaddlePaddle/CINN/pull/1133
 */
void cinn_call_cholesky_host(
    void* v_args, int num_args, int batch_size, int m, bool upper) {
#ifdef CINN_WITH_MKL_CBLAS
  cinn_pod_value_t* args = static_cast<cinn_pod_value_t*>(v_args);

  cinn_buffer_t* x = args[0].operator cinn_buffer_t*();
  cinn_buffer_t* out = args[1].operator cinn_buffer_t*();
  memcpy(out->memory, x->memory, x->memory_size);

  uint8_t bits = x->type.bits;
  PADDLE_ENFORCE_EQ(
      bits == 32 || bits == 64,
      true,
      ::common::errors::InvalidArgument(
          "Unsupported bits = %d float data type for cholesky.", bits));
  char uplo = upper ? 'U' : 'L';
  for (int i = 0; i < batch_size; i++) {
    if (bits == 32) {
      float* matrix = reinterpret_cast<float*>(out->memory) + i * m * m;
      LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, m, matrix, m);
    } else if (bits == 64) {
      double* matrix = reinterpret_cast<double*>(out->memory) + i * m * m;
      LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, m, matrix, m);
    }
  }
#else
  CINN_NOT_IMPLEMENTED
#endif
}

CINN_REGISTER_HELPER(cinn_cpu_mkl) {
  using namespace cinn;  // NOLINT
  using backends::FunctionProto;
  auto host_target = cinn::common::DefaultHostTarget();

  FunctionProto::shape_inference_t inference_shape_gemm =
      [](const std::vector<Expr>& args, int offset) {
        PADDLE_ENFORCE_EQ(
            offset, 0UL, ::common::errors::InvalidArgument("Only one output."));
        PADDLE_ENFORCE_EQ(args.size(),
                          12UL,
                          ::common::errors::InvalidArgument(
                              "Wrong number of arguments passed in."));
        auto M = cinn::optim::ArithSimplify(args[1]);
        auto N = cinn::optim::ArithSimplify(args[2]);
        std::vector<Expr> shape;
        shape.push_back(M);
        shape.push_back(N);
        return shape;
      };

  FunctionProto::shape_inference_t inference_shape_gemm_batch =
      [](const std::vector<Expr>& args, int offset) {
        PADDLE_ENFORCE_EQ(
            offset, 0UL, ::common::errors::InvalidArgument("Only one output."));
        PADDLE_ENFORCE_EQ(args.size(),
                          16UL,
                          ::common::errors::InvalidArgument(
                              "Wrong number of arguments passed in."));
        auto& A = args[14];
        auto A_tensor = A.as_tensor();
        PADDLE_ENFORCE_NOT_NULL(
            A_tensor,
            ::common::errors::InvalidArgument("expected type is tensor."));

        auto batch_size = cinn::optim::ArithSimplify(args[1]);
        int32_t batch_size_val = batch_size.as_int32();

        auto M = cinn::optim::ArithSimplify(args[2]);
        auto N = cinn::optim::ArithSimplify(args[3]);

        std::vector<Expr> shape;
        int total = 1;
        for (auto& v : A_tensor->shape) {
          auto val = cinn::optim::ArithSimplify(v);
          PADDLE_ENFORCE_EQ(
              val.is_constant(),
              true,
              ::common::errors::InvalidArgument("expected type is constant."));
          shape.push_back(val);
          total *= val.as_int32();
          if (total >= batch_size_val) break;
        }
        shape.push_back(M);
        shape.push_back(N);
        return shape;
      };

  REGISTER_EXTERN_FUNC_HELPER(cinn_cpu_mkl_gemm_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<float>()            // alpha
      .AddInputType<int>()              // M
      .AddInputType<int>()              // N
      .AddInputType<int>()              // K
      .AddInputType<bool>()             // ta
      .AddInputType<bool>()             // tb
      .AddInputType<int>()              // lda
      .AddInputType<int>()              // ldb
      .AddInputType<int>()              // ldc
      .AddInputType<float>()            // beta
      .AddInputType<cinn_buffer_t*>()   // A
      .AddInputType<cinn_buffer_t*>()   // B
      .AddOutputType<cinn_buffer_t*>()  // C
      .SetShapeInference(inference_shape_gemm)
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_cpu_mkl_gemm_batch_fp32, host_target)
      .SetRetType<void>()
      .AddInputType<float>()            // alpha
      .AddInputType<int>()              // batch
      .AddInputType<int>()              // M
      .AddInputType<int>()              // N
      .AddInputType<int>()              // K
      .AddInputType<bool>()             // ta
      .AddInputType<bool>()             // tb
      .AddInputType<int>()              // lda
      .AddInputType<int>()              // ldb
      .AddInputType<int>()              // ldc
      .AddInputType<int>()              // a_stride
      .AddInputType<int>()              // b_stride
      .AddInputType<int>()              // c_stride
      .AddInputType<float>()            // beta
      .AddInputType<cinn_buffer_t*>()   // A
      .AddInputType<cinn_buffer_t*>()   // B
      .AddOutputType<cinn_buffer_t*>()  // C
      .SetShapeInference(inference_shape_gemm_batch)
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_call_cholesky_host, host_target)
      .SetRetType<void>()
      .AddInputType<void*>()  // v_args
      .AddInputType<int>()    // num_args
      .AddInputType<int>()    // batch_size
      .AddInputType<int>()    // m
      .AddInputType<bool>()   // upper
      .End();

  return true;
}
