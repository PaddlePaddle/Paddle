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

#include "paddle/phi/kernels/layer_norm_grad_kernel.h"

#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/layer_norm_util.h"

namespace phi {

template <typename T, typename Context>
void LayerNormGradKernel(const Context &dev_ctx,
                         const DenseTensor &x,
                         paddle::optional<const DenseTensor &> scale_opt,
                         paddle::optional<const DenseTensor &> bias_opt,
                         const DenseTensor &mean,
                         const DenseTensor &variance,
                         const DenseTensor &out_grad,
                         float epsilon,
                         int begin_norm_axis,
                         bool is_test,
                         DenseTensor *x_grad,
                         DenseTensor *scale_grad,
                         DenseTensor *bias_grad) {
  using U = paddle::operators::LayerNormParamType<T>;
  // d_x, d_scale, d_bias may be nullptr
  auto *d_x = x_grad;
  auto *d_scale = scale_grad;
  auto *d_bias = bias_grad;

  auto *scale = scale_opt.get_ptr();
  auto *bias = bias_opt.get_ptr();
  auto *d_y = &out_grad;

  const auto &x_dims = x.dims();
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);

  auto *x_data = x.data<T>();
  auto *d_y_data = d_y->data<T>();

  auto *mean_data = mean.data<U>();
  auto *var_data = variance.data<U>();

  auto *d_x_data = (d_x == nullptr ? nullptr : dev_ctx.template Alloc<T>(d_x));

  auto x_dtype = x.dtype();

  phi::DataType scale_bias_dtype;
  if (scale != nullptr) {
    scale_bias_dtype = scale->dtype();
  } else {
    // FIXME(zengjinle): do not find a better way to get the right
    // data type of the d_scale and d_bias if scale == nullptr.
    if (bias != nullptr) {
      scale_bias_dtype = bias->dtype();
    } else {
      scale_bias_dtype = x_dtype;
    }
  }

#define PADDLE_LAUNCH_LAYERNORM_BWD(ScaleBiasT, IsScaleBiasSameDTypeWithX)  \
  do {                                                                      \
    auto *scale_data =                                                      \
        (scale == nullptr ? nullptr : scale->data<ScaleBiasT>());           \
    auto *d_scale_data =                                                    \
        (d_scale == nullptr ? nullptr                                       \
                            : dev_ctx.template Alloc<ScaleBiasT>(d_scale)); \
    auto *d_bias_data =                                                     \
        (d_bias == nullptr ? nullptr                                        \
                           : dev_ctx.template Alloc<ScaleBiasT>(d_bias));   \
    auto *d_x_data =                                                        \
        (d_x == nullptr ? nullptr : dev_ctx.template Alloc<T>(d_x));        \
    paddle::operators::LayerNormBackward<T, U, IsScaleBiasSameDTypeWithX>(  \
        x_data,                                                             \
        d_y_data,                                                           \
        scale_data,                                                         \
        mean_data,                                                          \
        var_data,                                                           \
        d_x_data,                                                           \
        d_scale_data,                                                       \
        d_bias_data,                                                        \
        epsilon,                                                            \
        batch_size,                                                         \
        feature_size,                                                       \
        dev_ctx);                                                           \
  } while (0)

  if (scale_bias_dtype == x_dtype) {
    PADDLE_LAUNCH_LAYERNORM_BWD(T, true);
  } else {
    PADDLE_LAUNCH_LAYERNORM_BWD(U, false);
  }

#undef PADDLE_LAUNCH_LAYERNORM_BWD
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(layer_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormGradKernel,
                   float,
                   phi::dtype::float16) {}
#elif CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(layer_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(layer_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
