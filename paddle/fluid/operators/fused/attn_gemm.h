/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/fluid/operators/fused/cublaslt.h"

namespace paddle {
namespace operators {

#if CUDA_VERSION >= 11060
// Only Used in Inference
template <typename T>
class CublasFusedMLP {
 public:
  // (m, n, k) = bsz_seq, hidden_feature, in_feature
  explicit CublasFusedMLP(const phi::GPUContext &dev_ctx) : dev_ctx_(dev_ctx) {
    if (std::is_same<T, paddle::platform::float16>::value) {
      mat_type_ = CUDA_R_16F;
      if (FLAGS_gemm_use_half_precision_compute_type) {
        // This option default value is true, it tends to result NaN, but get
        // better inference speed. you can turn off by using `export
        // FLAGS_gemm_use_half_precision_compute_type=0`.
        compute_type_ = CUBLAS_COMPUTE_16F;
        scale_type_ = CUDA_R_16F;
      }
    }
    if (std::is_same<T, platform::bfloat16>::value) {
      mat_type_ = CUDA_R_16BF;
    }
    if (std::is_same<T, double>::value) {
      mat_type_ = CUDA_R_64F;
      scale_type_ = CUDA_R_64F;
      compute_type_ = CUBLAS_COMPUTE_64F;
    }

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescCreate(
        &operation_desc_, compute_type_, scale_type_));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &a_desc_, mat_type_, 1, 1, 1));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &b_desc_, mat_type_, 1, 1, 1));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &out_desc_, mat_type_, 1, 1, 1));
  }
  ~CublasFusedMLP() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescDestroy(operation_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutDestroy(a_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutDestroy(b_desc_));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutDestroy(out_desc_));
  }

  void Setup(const phi::DDim &x_shape,
             const phi::DDim &w_shape,
             bool trans_x,
             bool trans_w) {
    // int64_t M = trans_x ? x_shape[1] : x_shape[0];
    // int64_t K = trans_w ? w_shape[1] : w_shape[0];
    // int64_t N = trans_w ? w_shape[0] : w_shape[1];
    // M_ = M;
    // K_ = K;
    // N_ = N;

    M_ = trans_x ? x_shape[1] : x_shape[0];
    K_ = trans_w ? w_shape[1] : w_shape[0];
    N_ = trans_w ? w_shape[0] : w_shape[1];

    // size_t m = 0, n = 0, k = 0;
    if (!trans_x) {
      M_ = x_shape[0];
      K_ = x_shape[1];
      cublas_ldb_ = K_;
    } else {
      M_ = x_shape[1];
      K_ = x_shape[0];
      cublas_ldb_ = M_;
    } 

    if (!trans_w) {
      N_ = w_shape[1];
      cublas_lda_ = N_;
    } else {
      N_ = w_shape[0];
      cublas_lda_ = K_;
    } 

    // cublas_M_ = n;
    // cublas_N_ = m;
    // cublas_K_ = k;
    // cublas_ldc_ = n;
    printf("M is: %ld, N is: %ld, K is: %ld \n", M_, N_, K_); 
    cublas_M_ = N_;
    cublas_N_ = M_;
    cublas_K_ = K_;
    cublas_ldc_ = N_;

    cublasOperation_t cublas_transA = trans_w ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublas_transB = trans_x ? CUBLAS_OP_T : CUBLAS_OP_N;

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc_,
            CUBLASLT_MATMUL_DESC_TRANSA,
            &cublas_transA,
            sizeof(cublas_transA)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc_,
            CUBLASLT_MATMUL_DESC_TRANSB,
            &cublas_transB,
            sizeof(cublas_transB)));

    SetCublasMatrixLayout(a_desc_, cublas_transA, cublas_M_, cublas_K_, cublas_lda_);
    SetCublasMatrixLayout(b_desc_, cublas_transB, cublas_K_, cublas_N_, cublas_ldb_);
    SetCublasMatrixLayout(out_desc_, CUBLAS_OP_N, cublas_M_, cublas_N_, cublas_ldc_);
  }

  void ComputeForward(const phi::DenseTensor *x,
                      const phi::DenseTensor *weight,
                      const phi::DenseTensor *bias,
                      phi::DenseTensor *residual,
                      phi::DenseTensor *output,
                      const std::string &activation) {
    T *out_data = output->data<T>();

    const bool add_residual = (residual == nullptr) ? false : true;
    const bool add_bias = (bias == nullptr) ? false : true;

    const T *bias_data = nullptr;
    if (add_bias) {
      printf("Here add bias \n"); 
      bias_data = bias->data<T>();
    }
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc_,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_data,
            sizeof(bias_data)));

    cublasLtEpilogue_t epiloque_func = GetEpilogueType(activation, add_bias);
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulDescSetAttribute(
            operation_desc_,
            CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epiloque_func,
            sizeof(epiloque_func)));

    T *residual_data = add_residual ? residual->data<T>() : out_data;

    cublasLtHandle_t lt_handle = dev_ctx_.cublaslt_handle();
    size_t workspace_size = static_cast<size_t>(4) * 1024 * 1024;
    cudaStream_t stream = dev_ctx_.stream();
    memory::allocation::AllocationPtr workspace = memory::Alloc(
        dev_ctx_.GetPlace(),
        workspace_size,
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx_.stream())));

    // if add_residual, we compute result + 1.0 * residual,
    // else result + 0.0 * out.
    double alpha64 = 1.0, beta64 = add_residual ? 1.0 : 0.0;
    float alpha32 = 1.0f, beta32 = add_residual ? 1.0f : 0.0f;
    half alpha16 = static_cast<half>(1.0),
         beta16 =
             add_residual ? static_cast<half>(1.0) : static_cast<half>(0.0);

    void *alpha = &alpha32, *beta = &beta32;
    if (std::is_same<T, double>::value) {
      alpha = &alpha64;
      beta = &beta64;
    }

    if (std::is_same<T, phi::dtype::float16>::value &&
        FLAGS_gemm_use_half_precision_compute_type) {
      alpha = &alpha16;
      beta = &beta16;
    }

    const auto *x_data = x->data<T>();
    const auto *w_data = weight->data<T>();

    cublasLtMatmulAlgo_t *algo =
        CublasLtAlgoCache::Instance().CublasLtAlgoSelect(lt_handle,
                                                         M_,
                                                         N_,
                                                         K_,
                                                         w_data,
                                                         x_data,
                                                         out_data,
                                                         &alpha,
                                                         &beta,
                                                         operation_desc_,
                                                         a_desc_,
                                                         b_desc_,
                                                         out_desc_,
                                                         compute_type_,
                                                         scale_type_,
                                                         mat_type_,
                                                         mat_type_,
                                                         mat_type_,
                                                         workspace->ptr(),
                                                         workspace_size,
                                                         stream);

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmul(lt_handle,
                                          operation_desc_,
                                          alpha,
                                          w_data,
                                          a_desc_,
                                          x_data,
                                          b_desc_,
                                          beta,
                                          residual_data,
                                          out_desc_,
                                          out_data,
                                          out_desc_,
                                          algo,
                                          workspace->ptr(),
                                          workspace_size,
                                          stream));
  }

 private:
  cublasLtEpilogue_t GetEpilogueType(const std::string &activation,
                                     const bool add_bias) {
    if (activation == "relu") {
      if (add_bias) {
        return CUBLASLT_EPILOGUE_RELU_BIAS;
      } else {
        return CUBLASLT_EPILOGUE_RELU;
      }
    } else if (activation == "gelu") {
      if (add_bias) {
        return CUBLASLT_EPILOGUE_GELU_BIAS;
      } else {
        return CUBLASLT_EPILOGUE_GELU;
      }
    } else if (activation == "none") {
      if (add_bias) {
        return CUBLASLT_EPILOGUE_BIAS;
      } else {
        return CUBLASLT_EPILOGUE_DEFAULT;
      }
    } else {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          platform::errors::InvalidArgument(
              "The activation attribute of fused_gemm_epilogue op should be"
              " one of {\"none\", \"relu\", \"gelu\"}. But received %s."
              "But received activation=%s.",
              activation));
    }
  }

  void SetCublasMatrixLayout(cublasLtMatrixLayout_t layout_desc,
                             cublasOperation_t cublas_trans,
                             const uint64_t cublas_row,
                             const uint64_t cublas_col, 
                             const uint64_t cublas_ld) {
    cudaDataType_t mat_type = CUDA_R_32F;
    if (std::is_same<T, paddle::platform::float16>::value) {
      mat_type = CUDA_R_16F;
    }
    if (std::is_same<T, platform::bfloat16>::value) {
      mat_type = CUDA_R_16BF;
    }
    if (std::is_same<T, double>::value) {
      mat_type = CUDA_R_64F;
    }

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutSetAttribute(
            layout_desc,
            CUBLASLT_MATRIX_LAYOUT_TYPE,
            &mat_type,
            sizeof(mat_type)));

    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutSetAttribute(
            layout_desc,
            CUBLASLT_MATRIX_LAYOUT_ROWS,
            cublas_trans == CUBLAS_OP_N ? &cublas_row : &cublas_col,
            sizeof(cublas_row)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutSetAttribute(
            layout_desc,
            CUBLASLT_MATRIX_LAYOUT_COLS,
            cublas_trans == CUBLAS_OP_N ? &cublas_col : &cublas_row,
            sizeof(cublas_col)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutSetAttribute(
            layout_desc,
            CUBLASLT_MATRIX_LAYOUT_LD,
            &cublas_ld,
            sizeof(cublas_ld)));
  }

  const phi::GPUContext &dev_ctx_;
  cublasLtMatmulDesc_t operation_desc_ = NULL;
  cublasLtMatrixLayout_t a_desc_ = NULL;
  cublasLtMatrixLayout_t b_desc_ = NULL;
  cublasLtMatrixLayout_t out_desc_ = NULL;
  int64_t M_ = 0;
  int64_t N_ = 0;
  int64_t K_ = 0;

  int64_t cublas_M_ = 0;
  int64_t cublas_N_ = 0;
  int64_t cublas_K_ = 0;
  int64_t cublas_lda_ = 0;
  int64_t cublas_ldb_ = 0;
  int64_t cublas_ldc_ = 0;

  cudaDataType_t mat_type_ = CUDA_R_32F;
  cudaDataType_t scale_type_ = CUDA_R_32F;
  cublasComputeType_t compute_type_ = CUBLAS_COMPUTE_32F;
};

#endif  // CUDA_VERSION >= 11060

// support gemm-nt and gemm-nn, which is used in fused_attention_op.
template <typename T>
class AttnMatMul {
 public:
  // (m, n, k) = bsz_seq, output_size, input_size
  AttnMatMul(const phi::GPUContext& dev_ctx,
             bool transA,
             bool transB,
             int bsz_seq,
             int output_size,
             int input_size,
             bool compute_bias)
      : dev_ctx_(dev_ctx),
        transA_(transA),
        transB_(transB),
        bsz_seq_(bsz_seq),
        output_size_(output_size),
        input_size_(input_size),
        compute_bias_(compute_bias) {}

  ~AttnMatMul() {}

  void ComputeForward(const phi::DenseTensor* weight,
                      const phi::DenseTensor* input,
                      const phi::DenseTensor* bias,
                      phi::DenseTensor* output,
                      phi::DenseTensor* bias_out) {
// #if CUDA_VERSION < 11060
    // // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
    // // here: (transa, transb): nt, input * weight.
    // CBLAS_TRANSPOSE transA = transA_ ? CblasTrans : CblasNoTrans;
    // CBLAS_TRANSPOSE transB = transB_ ? CblasTrans : CblasNoTrans;
    // T alpha = static_cast<T>(1.0);
    // T beta = static_cast<T>(0.0);

    // // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
    // auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    // blas.GEMM(transA,
    //           transB,
    //           bsz_seq_,
    //           output_size_,
    //           input_size_,
    //           alpha,
    //           input->data<T>(),
    //           weight->data<T>(),
    //           beta,
    //           output->data<T>());
    // if (compute_bias_) {
    //   // bias_out = output + bias
    //   std::vector<const phi::DenseTensor*> ins = {output, bias};
    //   std::vector<phi::DenseTensor*> outs = {bias_out};
    //   phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
    //       dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
    // }
// #else 
//     printf("=== Here cublaslt tokennum is: %d, output_size is: %d, input_size is: %d \n", bsz_seq_, output_size_, input_size_); 
    auto cublas_lt_gemm = CublasFusedMLP<T>(dev_ctx_); 
    phi::DDim input_shape({bsz_seq_, input_size_});
    phi::DDim weight_shape({input_size_, output_size_});
    if(transA_){
      input_shape[0] = input_size_; 
      input_shape[1] = bsz_seq_; 
    }
    if(transB_){
      weight_shape[0] = output_size_; 
      weight_shape[1] = input_size_; 
    }

    cublas_lt_gemm.Setup(input_shape, weight_shape, transA_, transB_); 
    if(compute_bias_){
      printf("Here use bias \n"); 
      cublas_lt_gemm.ComputeForward(input, weight, bias, nullptr, output, "none"); 
    } else {
      printf("Here without bias \n"); 
      cublas_lt_gemm.ComputeForward(input, weight, nullptr, nullptr, output, "none"); 
    }
// #endif // __CUDA_VERSION__ < 11060
  }

  void ComputeBackward(const phi::DenseTensor* input,
                       const phi::DenseTensor* weight,
                       const phi::DenseTensor* d_output,
                       phi::DenseTensor* d_input,
                       phi::DenseTensor* d_weight,
                       phi::DenseTensor* d_bias,
                       bool use_addto = false) {
    T alpha = static_cast<T>(1.0);
    T beta_dA = use_addto ? static_cast<T>(1.0) : static_cast<T>(0.0);
    T beta_dB = static_cast<T>(0.0);

    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    if (!transA_) {
      // forward: gemm-nt
      if (transB_) {
        // backward: gemm-tn, dB = (dC)^T * A
        if (d_weight) {
          int dB_m = output_size_;
          int dB_n = input_size_;
          int dB_k = bsz_seq_;

          T* dB_output_ptr = d_weight->data<T>();
          blas.GEMM(CblasTrans,
                    CblasNoTrans,
                    dB_m,
                    dB_n,
                    dB_k,
                    alpha,
                    d_output->data<T>(),
                    input->data<T>(),
                    beta_dB,
                    dB_output_ptr);
        }

        // backward: gemm-nn, dA = dC * B
        if (d_input) {
          int dA_m = bsz_seq_;
          int dA_n = input_size_;
          int dA_k = output_size_;

          T* dA_output_ptr = d_input->data<T>();
          blas.GEMM(CblasNoTrans,
                    CblasNoTrans,
                    dA_m,
                    dA_n,
                    dA_k,
                    alpha,
                    d_output->data<T>(),
                    weight->data<T>(),
                    beta_dA,
                    dA_output_ptr);
        }
      } else {  // fw: gemm-nn
        // backward: gemm-tn, dB = A^T * dC
        if (d_weight) {
          int dB_m = input_size_;
          int dB_n = output_size_;
          int dB_k = bsz_seq_;

          T* dB_output_ptr = d_weight->data<T>();
          blas.GEMM(CblasTrans,
                    CblasNoTrans,
                    dB_m,
                    dB_n,
                    dB_k,
                    alpha,
                    input->data<T>(),
                    d_output->data<T>(),
                    beta_dB,
                    dB_output_ptr);
        }

        // backward: gemm-nt, dA = dC * B^T
        if (d_input) {
          int dA_m = bsz_seq_;
          int dA_n = input_size_;
          int dA_k = output_size_;

          T* dA_output_ptr = d_input->data<T>();
          blas.GEMM(CblasNoTrans,
                    CblasTrans,
                    dA_m,
                    dA_n,
                    dA_k,
                    alpha,
                    d_output->data<T>(),
                    weight->data<T>(),
                    beta_dA,
                    dA_output_ptr);
        }
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "AttnMatMul wrapper do not support (transA=T, transB=T/N)"
          "parameters."));
    }
    if (compute_bias_ && d_bias) {
      // reduce: {0, 1, 2, 3, 4} -> {2, 3, 4} or {0, 1, 2} -> {2} or {0,1,2,3}
      // -> {3} or {0,1,2,3,4} -> {3,4}
      const auto input_dims = d_output->dims();
      const auto output_dims = d_bias->dims();
      bool support_case_1 =
          (input_dims.size() == 5 && output_dims.size() == 3 &&
           (input_dims[2] == output_dims[0]) &&
           (input_dims[3] == output_dims[1]) &&
           (input_dims[4] == output_dims[2]));
      bool support_case_2 =
          (input_dims.size() == 3 && output_dims.size() == 1 &&
           (input_dims[2] == output_dims[0]));
      bool support_case_3 =
          (input_dims.size() == 4 && output_dims.size() == 1 &&
           input_dims[3] == output_dims[0]);
      bool support_case_4 =
          (input_dims.size() == 5 && output_dims.size() == 2 &&
           input_dims[3] == output_dims[0] && input_dims[4] == output_dims[1]);

      gpuStream_t stream = dev_ctx_.stream();
      if (support_case_1 || support_case_2) {
        TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
            dev_ctx_,
            *d_output,
            d_bias,
            kps::IdentityFunctor<T>(),
            {0, 1},
            stream);
      } else if (support_case_3 || support_case_4) {
        TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
            dev_ctx_,
            *d_output,
            d_bias,
            kps::IdentityFunctor<T>(),
            {0, 1, 2},
            stream);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Only support reduce when the input dims are [0,1,2,3,4] and "
            "output is [2,3,4]"
            "or input is [0,1,2] and output is [2]."));
      }
    }
  }

 private:
  const phi::GPUContext& dev_ctx_;

  bool transA_;
  bool transB_;

  int bsz_seq_;
  int output_size_;
  int input_size_;

  int compute_bias_;
};


}  // namespace operators
}  // namespace paddle
