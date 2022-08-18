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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
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

  void ComputeForward(const framework::Tensor* weight,
                      const framework::Tensor* input,
                      const framework::Tensor* bias,
                      framework::Tensor* output,
                      framework::Tensor* bias_out) {
    // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
    // here: (transa, transb): nt, input * weight.
    CBLAS_TRANSPOSE transA = transA_ ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = transB_ ? CblasTrans : CblasNoTrans;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    // (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    blas.GEMM(transA,
              transB,
              bsz_seq_,
              output_size_,
              input_size_,
              alpha,
              input->data<T>(),
              weight->data<T>(),
              beta,
              output->data<T>());
    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const Tensor*> ins = {output, bias};
      std::vector<Tensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
    }
  }

  void ComputeBackward(const framework::Tensor* input,
                       const framework::Tensor* weight,
                       const framework::Tensor* d_output,
                       framework::Tensor* d_input,
                       framework::Tensor* d_weight,
                       framework::Tensor* d_bias,
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
