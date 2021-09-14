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

#include "paddle/fluid/operators/fused/attn_bias_add.cu.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/float16.h"

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

  // todo: parameter order
  void ComputeForward(const T* weight_data, const T* input_data,
                      const T* bias_data, T* output_data, T* bias_out_data) {
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
              input_data, weight_data, beta, output_data);
    if (compute_bias_) {
      // compute output + bias
      LaunchBiasAddFwKernel(dev_ctx_, bsz_seq_, output_size_, output_data,
                            bias_data, bias_out_data);
    }
  }

  void ComputeBackward(const T* input, const T* weight, const T* d_output,
                       T* d_input, T* d_weight, T* d_bias) {
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
    T* dB_output_ptr = d_weight;

    T* dA_input_1_ptr = nullptr;
    T* dA_input_2_ptr = nullptr;
    T* dA_output_ptr = d_input;

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

        blas.GEMM(dB_transA, dB_transB, dB_m, dB_n, dB_k, alpha, d_output,
                  input, beta, dB_output_ptr);
        blas.GEMM(dA_transA, dA_transB, dA_m, dA_n, dA_k, alpha, d_output,
                  weight, beta, dA_output_ptr);
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

        blas.GEMM(dB_transA, dB_transB, dB_m, dB_n, dB_k, alpha, input,
                  d_output, beta, dB_output_ptr);
        blas.GEMM(dA_transA, dA_transB, dA_m, dA_n, dA_k, alpha, d_output,
                  weight, beta, dA_output_ptr);
      }
    } else if (transB_) {
      // todo: not support
    } else {
      // todo: not support
    }
    if (compute_bias_) {
      LaunchBiasAddBwKernel(dev_ctx_, bsz_seq_, output_size_, d_output, d_bias);
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
