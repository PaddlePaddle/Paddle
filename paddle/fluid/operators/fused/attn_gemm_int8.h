/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <iostream>
#include <vector>
#include "paddle/fluid/operators/fused/cublaslt.h"
#include "paddle/fluid/operators/fused/quant_dequant_kernel.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace paddle {
namespace operators {

using phi::backends::gpu::GpuLaunchConfig;

template <typename T>
class AttnMatmulINT8 {
 public:
  AttnMatmulINT8(
      const phi::GPUContext& dev_ctx, int m, int n, int k, bool compute_bias)
      : dev_ctx_(dev_ctx), m_(m), n_(n), k_(k), compute_bias_(compute_bias) {
    auto helper = std::make_shared<CublasLtHelper>(m, k, n);
    helpers_.emplace_back(helper);
    gpu_config_ = std::make_unique<GpuLaunchConfig>(
        phi::backends::gpu::GetGpuLaunchConfig1D(
            dev_ctx, m * n, DequantKernelVecSize));
  }
  ~AttnMatmulINT8() {}

  // This function is used to execute GEMM, with input and output's types are
  // both T.
  void ComputeForward(const phi::DenseTensor* weight,
                      const phi::DenseTensor* input,
                      phi::DenseTensor* input_tmp,
                      const phi::DenseTensor* bias,
                      phi::DenseTensor* output,
                      phi::DenseTensor* output_tmp,
                      phi::DenseTensor* bias_out,
                      const float quant_in_scale,
                      const phi::DenseTensor* dequant_out_scale,
                      const int quant_round_type = 1,
                      const float quant_max_bound = 127.0,
                      const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());

    helpers_[0]->GEMM(input_tmp->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  gpu_config_.get(),
                                  quant_in_scale,
                                  dequant_out_scale->data<float>());

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(cudaGetLastError(),
                        cudaSuccess,
                        platform::errors::Fatal(
                            "cuda error occured after computing bias. "
                            "But it does not mean this error is caused by "
                            "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are
  // both INT8.
  void ComputeForwardINT8ToINT8(const phi::DenseTensor* weight,
                                phi::DenseTensor* input,
                                const phi::DenseTensor* bias,
                                phi::DenseTensor* output,
                                phi::DenseTensor* bias_out,
                                void* workspace = nullptr) {
    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output->data<int32_t>(),
                      dev_ctx_.stream(),
                      workspace);
  }

  // This function is used to execute GEMM, with input and output's types are
  // INT8 and T.
  void ComputeForwardINT8ToT(const phi::DenseTensor* weight,
                             const float quant_in_scale,
                             phi::DenseTensor* input,
                             const phi::DenseTensor* bias,
                             phi::DenseTensor* output,
                             phi::DenseTensor* output_tmp,
                             phi::DenseTensor* bias_out,
                             const phi::DenseTensor* dequant_out_scale) {
    helpers_[0]->GEMM(input->data<int8_t>(),
                      weight->data<int8_t>(),
                      output_tmp->data<int32_t>(),
                      dev_ctx_.stream());

    dequantize_kernel_launcher<T>(output_tmp->data<int32_t>(),
                                  output->data<T>(),
                                  m_,
                                  n_,
                                  dev_ctx_.stream(),
                                  gpu_config_.get(),
                                  quant_in_scale,
                                  dequant_out_scale->data<float>());

    if (compute_bias_) {
      // bias_out = output + bias
      std::vector<const phi::DenseTensor*> ins = {output, bias};
      std::vector<phi::DenseTensor*> outs = {bias_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
      PADDLE_ENFORCE_EQ(cudaGetLastError(),
                        cudaSuccess,
                        platform::errors::Fatal(
                            "cuda error occured after computing bias. "
                            "But it does not mean this error is caused by "
                            "bias computing"));
    }
  }

  // This function is used to execute GEMM, with input and output's types are T
  // and INT8.
  void ComputeForwardTToINT8(const phi::DenseTensor* weight,
                             const float quant_in_scale,
                             const phi::DenseTensor* input,
                             phi::DenseTensor* input_tmp,
                             const phi::DenseTensor* bias,
                             phi::DenseTensor* output,
                             phi::DenseTensor* bias_out,
                             const int quant_round_type = 1,
                             const float quant_max_bound = 127.0,
                             const float quant_min_bound = -127.0) {
    quantize_kernel_launcher<T>(input->data<T>(),
                                input_tmp->data<int8_t>(),
                                quant_in_scale,
                                m_,
                                k_,
                                quant_round_type,
                                quant_max_bound,
                                quant_min_bound,
                                dev_ctx_.stream());

    helpers_[0]->GEMM(input_tmp->data<int8_t>(),
                      weight->data<int8_t>(),
                      output->data<int32_t>(),
                      dev_ctx_.stream());
  }

 private:
  const phi::GPUContext& dev_ctx_;

  int m_;  // m
  int n_;  // n
  int k_;  // k

  int compute_bias_;
  std::vector<std::shared_ptr<CublasLtHelper>> helpers_;
  std::unique_ptr<GpuLaunchConfig> gpu_config_;
};

}  // namespace operators
}  // namespace paddle
