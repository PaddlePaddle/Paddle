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

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/float16.h"

#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"

namespace paddle {
namespace operators {

// support gemm-nt and gemm-nn, which is used in fused_attention_op.
template <typename T>
class AttnMatMul {
 public:
  // (m, n, k) = bsz_seq, output_size, input_size
  AttnMatMul(const platform::CUDADeviceContext& dev_ctx, bool transA,
             bool transB, int bsz_seq, int output_size, int input_size,
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
                      const framework::Tensor* bias, framework::Tensor* output,
                      framework::Tensor* bias_out) {
    // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
    // here: (transa, transb): nt, input * weight.
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    if (transA_) {
      transA = CblasTrans;
    }
    if (transB_) {
      transB = CblasTrans;
    }
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    // here: (m, n, k) = bsz_seq, output_size, input_size, (input, weight, out)
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    blas.GEMM(transA, transB, bsz_seq_, output_size_, input_size_, alpha,
              input->data<T>(), weight->data<T>(), beta, output->data<T>());
    if (compute_bias_) {
      // compute output + bias
      std::vector<const Tensor*> ins;
      std::vector<Tensor*> outs;
      ins.emplace_back(output);
      ins.emplace_back(bias);
      outs.emplace_back(bias_out);
      int elewise_add_axis = -1;
      LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, elewise_add_axis, AddFunctor<T>());
    }
  }

  void ComputeBackward(const framework::Tensor* input,
                       const framework::Tensor* weight,
                       const framework::Tensor* d_output,
                       framework::Tensor* d_input, framework::Tensor* d_weight,
                       framework::Tensor* d_bias) {
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);

    CBLAS_TRANSPOSE dB_transA = CblasNoTrans;
    CBLAS_TRANSPOSE dB_transB = CblasNoTrans;
    CBLAS_TRANSPOSE dA_transA = CblasNoTrans;
    CBLAS_TRANSPOSE dA_transB = CblasNoTrans;
    int dB_m = 1;
    int dB_n = 1;
    int dB_k = 1;
    int dA_m = 1;
    int dA_n = 1;
    int dA_k = 1;

    T* dB_input_1_ptr = nullptr;
    T* dB_input_2_ptr = nullptr;
    T* dB_output_ptr = d_weight->data<T>();

    T* dA_input_1_ptr = nullptr;
    T* dA_input_2_ptr = nullptr;
    T* dA_output_ptr = d_input->data<T>();

    if (!transA_) {
      // fw: gemm-nt
      if (transB_) {
        // bw: gemm-tn, dB = (dC)^t * A
        dB_transA = CblasTrans;
        dB_transB = CblasNoTrans;
        dB_m = output_size_;
        dB_n = input_size_;
        dB_k = bsz_seq_;

        // bw: gemm-nn, dA = dC * B
        dA_transA = CblasNoTrans;
        dA_transB = CblasNoTrans;
        dA_m = bsz_seq_;
        dA_n = input_size_;
        dA_k = output_size_;

        blas.GEMM(dB_transA, dB_transB, dB_m, dB_n, dB_k, alpha,
                  d_output->data<T>(), input->data<T>(), beta, dB_output_ptr);
        blas.GEMM(dA_transA, dA_transB, dA_m, dA_n, dA_k, alpha,
                  d_output->data<T>(), weight->data<T>(), beta, dA_output_ptr);
      } else {  // fw: gemm-nn
        // bw: gemm-tn, dB = A^t * dC
        dB_transA = CblasTrans;
        dB_transB = CblasNoTrans;
        dB_m = input_size_;
        dB_n = output_size_;
        dB_k = bsz_seq_;

        // bw: gemm-nt, dA = dC * B^t
        dA_transA = CblasNoTrans;
        dA_transB = CblasTrans;
        dA_m = bsz_seq_;
        dA_n = input_size_;
        dA_k = output_size_;

        blas.GEMM(dB_transA, dB_transB, dB_m, dB_n, dB_k, alpha,
                  input->data<T>(), d_output->data<T>(), beta, dB_output_ptr);
        blas.GEMM(dA_transA, dA_transB, dA_m, dA_n, dA_k, alpha,
                  d_output->data<T>(), weight->data<T>(), beta, dA_output_ptr);
      }
    } else if (transB_) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "AttnMatMul wrapper do not support (transA=T, transB=T)"
          "parameters."));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "AttnMatMul wrapper do not support (transA=T, transB=N)"
          "parameters."));
    }
    if (compute_bias_) {
      // reduce: {0, 1, 2, 3, 4} -> {2, 3, 4} or {0, 1, 2} -> {2}
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
      if (support_case_1 || support_case_2) {
        gpuStream_t stream = dev_ctx_.stream();
        TensorReduceFunctorImpl<T, T, CustomSum>(*d_output, d_bias, {0, 1},
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
  const platform::CUDADeviceContext& dev_ctx_;

  bool transA_;
  bool transB_;

  int bsz_seq_;
  int output_size_;
  int input_size_;

  int compute_bias_;
};

}  // namespace operators
}  // namespace paddle
