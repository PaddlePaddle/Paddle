// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/batch_norm_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void BatchNormGradRawKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& bias,
                            const paddle::optional<DenseTensor>& mean,
                            const paddle::optional<DenseTensor>& variance,
                            const DenseTensor& saved_mean,
                            const DenseTensor& saved_variance,
                            const paddle::optional<DenseTensor>& reserve_space,
                            const DenseTensor& y_grad,
                            float momentum,
                            float epsilon,
                            const std::string& data_layout,
                            bool is_test,
                            bool use_global_stats,
                            bool trainable_statistics,
                            bool is_inplace,
                            DenseTensor* x_grad,
                            DenseTensor* scale_grad,
                            DenseTensor* bias_grad) {
  funcs::BatchNormOneDNNHandler<T> handler(
      dev_ctx.GetEngine(), dev_ctx.GetPlace(), epsilon, &x, &scale, &y_grad);

  const unsigned int C = vectorize(scale.dims())[0];
  const size_t scaleshift_size = 2 * C;
  std::vector<T> diff_scaleshift_data;
  diff_scaleshift_data.reserve(scaleshift_size);

  T* diff_scale_data = dev_ctx.template Alloc<T>(scale_grad);
  T* diff_shift_data = dev_ctx.template Alloc<T>(bias_grad);

  auto src_memory = handler.AcquireSrcMemory(&x);
  auto mean_memory = handler.AcquireMeanMemory(&saved_mean);
  auto variance_memory = handler.AcquireVarianceMemory(&saved_variance);
  auto diff_dst_memory = handler.AcquireDiffDstMemory(&y_grad);
  auto scaleshift_mems = handler.AcquireScaleShiftMemory(&scale, &bias);
  auto diff_src_memory = handler.AcquireDiffSrcMemory(x_grad);
  auto diff_scaleshift_mems =
      handler.AcquireDiffScaleShiftMemory(diff_scale_data, diff_shift_data);

  auto batch_norm_bwd_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  batch_norm_bwd_p->execute(
      astream,
      {{DNNL_ARG_SRC, *src_memory},
       {DNNL_ARG_MEAN, *mean_memory},
       {DNNL_ARG_VARIANCE, *variance_memory},
       {DNNL_ARG_DIFF_DST, *diff_dst_memory},
       {DNNL_ARG_SCALE, *(std::get<0>(scaleshift_mems))},
       {DNNL_ARG_SHIFT, *(std::get<1>(scaleshift_mems))},
       {DNNL_ARG_DIFF_SRC, *diff_src_memory},
       {DNNL_ARG_DIFF_SCALE, *(std::get<0>(diff_scaleshift_mems))},
       {DNNL_ARG_DIFF_SHIFT, *(std::get<1>(diff_scaleshift_mems))}});
  astream.wait();

  // copy back diff scale/shift to output tensors (diff scale/shift)
  diff_scaleshift_data.resize(scaleshift_size);
  auto it = std::begin(diff_scaleshift_data);
  std::copy(it, std::next(it, C), diff_scale_data);
  std::copy(std::next(it, C), std::end(diff_scaleshift_data), diff_shift_data);

  // set memory descriptor of out tensor
  x_grad->set_mem_desc(diff_src_memory->get_desc());
}

template <typename T, typename Context>
void BatchNormGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& scale,
                         const DenseTensor& bias,
                         const paddle::optional<DenseTensor>& mean,
                         const paddle::optional<DenseTensor>& variance,
                         const DenseTensor& saved_mean,
                         const DenseTensor& saved_variance,
                         const paddle::optional<DenseTensor>& reserve_space,
                         const DenseTensor& y_grad,
                         float momentum,
                         float epsilon,
                         const std::string& data_layout,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         DenseTensor* x_grad,
                         DenseTensor* scale_grad,
                         DenseTensor* bias_grad) {
  BatchNormGradRawKernel<T, Context>(dev_ctx,
                                     x,
                                     scale,
                                     bias,
                                     mean,
                                     variance,
                                     saved_mean,
                                     saved_variance,
                                     reserve_space,
                                     y_grad,
                                     momentum,
                                     epsilon,
                                     data_layout,
                                     is_test,
                                     use_global_stats,
                                     trainable_statistics,
                                     /*is_inplace*/ false,
                                     x_grad,
                                     scale_grad,
                                     bias_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    batch_norm_grad, OneDNN, ONEDNN, phi::BatchNormGradKernel, float) {}
PD_REGISTER_KERNEL(
    batch_norm_grad_raw, OneDNN, ONEDNN, phi::BatchNormGradRawKernel, float) {}
