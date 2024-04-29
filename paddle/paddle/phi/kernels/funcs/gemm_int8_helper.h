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

#include "Paddle/paddle/phi/kernels/funcs/cublaslt.h"

namespace phi {

template <typename T>
class Int8GEMMHelper {
 public:
  Int8GEMMHelper(const phi::GPUContext &dev_ctx,
                 int m,
                 int k,
                 int n,
                 phi::DenseTensor &workspace,        // NOLINT
                 phi::DenseTensor &input_workspace,  // NOLINT
                 phi::DenseTensor &out_workspace,    // NOLINT
                 int quant_round_type,
                 float quant_max_bound,
                 float quant_min_bound)
      : dev_ctx_(dev_ctx),
        m_(m),
        k_(k),
        n_(n),
        quant_round_type_(quant_round_type),
        quant_min_bound_(quant_min_bound),
        quant_max_bound_(quant_max_bound),
        workspace_(workspace),
        input_workspace_(input_workspace),
        out_workspace_(out_workspace) {
    cublaslt_helper = std::make_unique<CublasLtHelper<int32_t>>(
        m, k, n, dev_ctx.cublaslt_handle());
  }

  void Compute(const phi::DenseTensor *input,
               const phi::DenseTensor *weight,  // int8, Need be transposed
               const phi::DenseTensor *dequant_out_scales,
               const float quant_in_scale,
               phi::DenseTensor *output,
               bool quant_in = false,
               bool dequant_out = false) {
    phi::DenseTensor input_tmp, out_tmp;
    if (quant_in) {
      input_tmp = input_workspace_;
      LaunchQuantKernel<T>(input->data<T>(),
                           input_tmp.data<int8_t>(),
                           quant_in_scale,
                           m_,
                           k_,
                           quant_round_type_,
                           quant_max_bound_,
                           quant_min_bound_,
                           dev_ctx_.stream());
    } else {
      input_tmp = *input;
    }

    if (dequant_out) {
      out_tmp = out_workspace_;
    } else {
      out_tmp = *output;
    }

    cublaslt_helper->GEMM(input_tmp.data<int8_t>(),
                          weight->data<int8_t>(),
                          out_tmp.data<int32_t>(),
                          dev_ctx_.stream(),
                          (void *)workspace_.data<int8_t>(),
                          workspace_.numel());

    if (dequant_out) {
      auto gpu_config = std::make_unique<GpuLaunchConfig>(
          phi::backends::gpu::GetGpuLaunchConfig1D(
              dev_ctx_, m_ * n_, DequantKernelVecSize));
      LaunchDequantKernel<T>(out_tmp.data<int32_t>(),
                             output->data<T>(),
                             m_,
                             n_,
                             dev_ctx_.stream(),
                             gpu_config.get(),
                             quant_in_scale,
                             dequant_out_scales->data<float>());
    }
  }

 private:
  const phi::GPUContext &dev_ctx_;
  int m_;
  int k_;
  int n_;
  int quant_round_type_;
  float quant_max_bound_;
  float quant_min_bound_;
  phi::DenseTensor &workspace_;        // char
  phi::DenseTensor &input_workspace_;  // int8_t
  phi::DenseTensor &out_workspace_;    // int32_t

  std::unique_ptr<CublasLtHelper<int32_t>> cublaslt_helper;
};

}  // namespace phi
