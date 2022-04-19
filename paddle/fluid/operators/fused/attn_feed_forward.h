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
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

template <typename T>
class FeedForward {
 public:
  FeedForward(const platform::CUDADeviceContext& dev_ctx, int bsz_seq,
              int output_size, int input_size, bool compute_bias)
      : dev_ctx_(dev_ctx),
        bsz_seq_(bsz_seq),
        output_size_(output_size),
        input_size_(input_size),
        compute_bias_(compute_bias) {}

  ~FeedForward() {}

  void ComputeForward(const T* weight_data, const T* input_data,
                      const T* bias_data, T* output_data, T* bias_out_data) {
    // Note: for blas.GEMM API in Paddle, it treats all inputs as row-major.
    // To convert to col-major expression, transa<->transb, A<->B，m<->n.

    // column-major: gemm-tn.
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    // column-major: (m,n,k) = output_size,bsz_seq,input_size (weight*input=out)
    // here: (m,n,k) = bsz_seq,output_size,input_size (input*weight=out)
    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    blas.GEMM(transA, transB, bsz_seq_, output_size_, input_size_, alpha,
              input_data, weight_data, beta, output_data);
    if (compute_bias_) {
      LaunchBiasAddFwKernel(dev_ctx_, bsz_seq_, output_size_, output_data,
                            bias_data, bias_out_data);
    }
  }

  void ComputeBackward(T* input, T* weight, T* d_output, T* d_input,
                       T* d_weight, T* d_bias) {
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);

    // column-major: gemm-nt, get d_weight.
    CBLAS_TRANSPOSE transA = CblasTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    // column-major: (m,n,k): input_size,output_size,bsz (input*dout=dweight)
    // here: (m,n,k): output_size,input_size,bsz (dout*input=dweight)
    blas.GEMM(transA, transB, output_size_, input_size_, bsz_seq_, alpha,
              d_output, input, beta, d_weight);

    // column-major: gemm-nn: get d_input.
    transA = CblasNoTrans;
    // column-major: (m,n,k): input_size,bsz,output_size (weight*dout=dinput)
    // here: (m, n, k): bsz, input_size, output_size, (dout*weight=dinput)
    blas.GEMM(transA, transB, bsz_seq_, input_size_, output_size_, alpha,
              d_output, weight, beta, d_input);
    if (compute_bias_) {
      LaunchBiasAddBwKernel(dev_ctx_, bsz_seq_, output_size_, d_output, d_bias);
    }
  }

 private:
  const platform::CUDADeviceContext& dev_ctx_;
  int bsz_seq_, output_size_, input_size_;
  bool compute_bias_;
};

}  // namespace operators
}  // namespace paddle
