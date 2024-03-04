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

#define PD_DECLARE_BN_GRAD_FUNCTOR(dtype, backend)                         \
  template void phi::BatchNormGradFunctor<dtype, ::phi::backend##Context>( \
      const ::phi::backend##Context& dev_ctx,                              \
      const DenseTensor& x,                                                \
      const paddle::optional<DenseTensor>& scale,                          \
      const paddle::optional<DenseTensor>& bias,                           \
      const paddle::optional<DenseTensor>& mean,                           \
      const paddle::optional<DenseTensor>& variance,                       \
      const DenseTensor& saved_mean,                                       \
      const DenseTensor& saved_variance,                                   \
      const paddle::optional<DenseTensor>& reserve_space,                  \
      const DenseTensor& y_grad,                                           \
      float momentum,                                                      \
      float epsilon,                                                       \
      const std::string& data_layout,                                      \
      bool is_test,                                                        \
      bool use_global_stats,                                               \
      bool trainable_statistics,                                           \
      bool is_inplace,                                                     \
      DenseTensor* x_grad,                                                 \
      DenseTensor* scale_grad,                                             \
      DenseTensor* bias_grad)

namespace phi {

template <typename T, typename Context>
void BatchNormGradFunctor(const Context& dev_ctx,
                          const DenseTensor& x,
                          const paddle::optional<DenseTensor>& scale,
                          const paddle::optional<DenseTensor>& bias,
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
  auto Scale = scale.get_ptr();
  auto Bias = bias.get_ptr();
  const bool use_scale = scale ? true : false;
  const bool use_bias = bias ? true : false;

  std::vector<int64_t> scale_tz;
  std::vector<int64_t> bias_tz;
  if (use_scale) {
    scale_tz = common::vectorize<int64_t>(Scale->dims());
    PADDLE_ENFORCE_EQ(
        scale_tz.size(),
        1,
        errors::InvalidArgument(
            "Dims of scale tensor must be 1, but received scale's size is %d",
            scale_tz.size()));
  }
  if (use_bias) {
    bias_tz = common::vectorize<int64_t>(Bias->dims());
    PADDLE_ENFORCE_EQ(
        bias_tz.size(),
        1,
        errors::InvalidArgument(
            "Dims of bias tensor must be 1, but received bias's size is %d",
            bias_tz.size()));
  }

  funcs::BatchNormOneDNNHandler<T> handler(dev_ctx.GetEngine(),
                                           dev_ctx.GetPlace(),
                                           epsilon,
                                           &x,
                                           use_scale,
                                           use_bias,
                                           &y_grad);

  T* diff_scale_data = dev_ctx.template Alloc<T>(scale_grad);
  T* diff_shift_data = dev_ctx.template Alloc<T>(bias_grad);

  auto src_memory = handler.AcquireSrcMemory(&x);
  auto mean_memory = handler.AcquireMeanMemory(&saved_mean);
  auto variance_memory = handler.AcquireVarianceMemory(&saved_variance);
  auto diff_dst_memory = handler.AcquireDiffDstMemory(&y_grad);
  auto diff_src_memory = handler.AcquireDiffSrcMemory(x_grad);

  auto batch_norm_bwd_p = handler.AcquireBackwardPrimitive();

  std::shared_ptr<dnnl::memory> scale_memory(nullptr);
  std::shared_ptr<dnnl::memory> diff_scale_memory(nullptr);
  std::shared_ptr<dnnl::memory> diff_shift_memory(nullptr);
  if (scale) {
    scale_memory = handler.AcquireScaleMemory(Scale);
    diff_scale_memory = handler.AcquireDiffScaleMemory(diff_scale_data);
  }
  if (bias) diff_shift_memory = handler.AcquireDiffShiftMemory(diff_shift_data);

  auto& astream = OneDNNContext::tls().get_stream();
  batch_norm_bwd_p->execute(astream,
                            {{DNNL_ARG_SRC, *src_memory},
                             {DNNL_ARG_MEAN, *mean_memory},
                             {DNNL_ARG_VARIANCE, *variance_memory},
                             {DNNL_ARG_DIFF_DST, *diff_dst_memory},
                             {DNNL_ARG_SCALE, *scale_memory},
                             {DNNL_ARG_DIFF_SRC, *diff_src_memory},
                             {DNNL_ARG_DIFF_SCALE, *diff_scale_memory},
                             {DNNL_ARG_DIFF_SHIFT, *diff_shift_memory}});
  astream.wait();

  // set memory descriptor of out tensor
  x_grad->set_mem_desc(diff_src_memory->get_desc());
}

template <typename T, typename Context>
void BatchNormGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& scale,
                         const paddle::optional<DenseTensor>& bias,
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
  BatchNormGradFunctor<T, Context>(dev_ctx,
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

PD_DECLARE_BN_GRAD_FUNCTOR(float, OneDNN);

PD_REGISTER_KERNEL(
    batch_norm_grad, OneDNN, ONEDNN, phi::BatchNormGradKernel, float) {}
