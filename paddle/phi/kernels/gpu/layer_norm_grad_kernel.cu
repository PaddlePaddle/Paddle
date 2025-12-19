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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#include "paddle/phi/kernels/funcs/layer_norm_util.h"
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
#include "paddle/phi/kernels/funcs/fast_ln_v2.h"
#endif
namespace phi {
enum class LayerNormGadKernelVariant { FAST_LN_V2, GENERIC };
static inline LayerNormGadKernelVariant LayerNormGradKernelDispatch(
    const paddle::DataType weight_type,
    const paddle::DataType input_type,
    const paddle::DataType output_type,
    const paddle::DataType compute_type,
    const uint32_t hidden_size,
    const DenseTensor *scale) {
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
  if (scale != nullptr && input_type != paddle::DataType::FLOAT32 &&
      hidden_size != 4096 && hidden_size > 1024) {
    // using fast_ln_v2 only sm > 70
    auto prop = funcs::fast_ln_v2::GetDeviceProp();
    if (prop->major > 7 &&
        funcs::fast_ln_v2::has_fast_ln_v2_bwd_kernel(
            weight_type, input_type, output_type, compute_type, hidden_size)) {
      return LayerNormGadKernelVariant::FAST_LN_V2;
    }
  }
#endif
  return LayerNormGadKernelVariant::GENERIC;
}

template <typename T, typename Context>
void LayerNormGradKernel(const Context &dev_ctx,
                         const DenseTensor &x,
                         const paddle::optional<DenseTensor> &scale_opt,
                         const paddle::optional<DenseTensor> &bias_opt,
                         const DenseTensor &mean,
                         const DenseTensor &variance,
                         const DenseTensor &out_grad,
                         float epsilon,
                         int begin_norm_axis,
                         DenseTensor *x_grad,
                         DenseTensor *scale_grad,
                         DenseTensor *bias_grad) {
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    if (scale_grad) {
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(common::vectorize(scale_grad->dims())),
          0,
          scale_grad);
      if (scale_opt.get_ptr() && x.dtype() != scale_opt.get().dtype()) {
        phi::CastKernel<T, Context>(
            dev_ctx, *scale_grad, scale_opt.get().dtype(), scale_grad);
      }
    }
    if (bias_grad) {
      phi::Full<T, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(bias_grad->dims())),
                            0,
                            bias_grad);
      if (bias_opt.get_ptr() && x.dtype() != bias_opt.get().dtype()) {
        phi::CastKernel<T, Context>(
            dev_ctx, *bias_grad, bias_opt.get().dtype(), bias_grad);
      }
    }
    return;
  }
  using U = funcs::LayerNormParamType<T>;
  // d_x, d_scale, d_bias may be nullptr
  auto *d_x = x_grad;
  auto *d_scale = scale_grad;
  auto *d_bias = bias_grad;

  auto *scale = scale_opt.get_ptr();
  auto *bias = bias_opt.get_ptr();
  auto *d_y = &out_grad;

  const auto &x_dims = x.dims();
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
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
    funcs::LayerNormBackward<T, U, IsScaleBiasSameDTypeWithX>(x_data,       \
                                                              d_y_data,     \
                                                              scale_data,   \
                                                              mean_data,    \
                                                              var_data,     \
                                                              d_x_data,     \
                                                              d_scale_data, \
                                                              d_bias_data,  \
                                                              epsilon,      \
                                                              batch_size,   \
                                                              feature_size, \
                                                              dev_ctx);     \
  } while (0)

#define PADDLE_LAUNCH_FAST_LAYERNORM_V2_BWD(ScaleBiasT)                     \
  do {                                                                      \
    auto stream = dev_ctx.stream();                                         \
    auto place = x.place();                                                 \
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
    funcs::fast_ln_v2::LaunchNormBwd<T, Context>(dev_ctx,                   \
                                                 stream,                    \
                                                 place,                     \
                                                 x_data,                    \
                                                 scale_data,                \
                                                 mean_data,                 \
                                                 var_data,                  \
                                                 d_y_data,                  \
                                                 d_x_data,                  \
                                                 d_scale_data,              \
                                                 d_bias_data,               \
                                                 scale_bias_dtype,          \
                                                 x_dtype,                   \
                                                 x_grad->dtype(),           \
                                                 compute_dtype,             \
                                                 feature_size,              \
                                                 batch_size,                \
                                                 feature_size,              \
                                                 epsilon);                  \
  } while (0)

  auto compute_dtype = phi::CppTypeToDataType<U>::Type();
  auto kernel_variant = LayerNormGradKernelDispatch(
      scale_bias_dtype, x_dtype, x_dtype, compute_dtype, feature_size, scale);
  switch (kernel_variant) {
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
    case LayerNormGadKernelVariant::FAST_LN_V2:
      if (scale_bias_dtype == x_dtype) {
        PADDLE_LAUNCH_FAST_LAYERNORM_V2_BWD(T);
      } else {
        PADDLE_LAUNCH_FAST_LAYERNORM_V2_BWD(U);
      }
      break;
#endif
    case LayerNormGadKernelVariant::GENERIC:
    default:
      if (scale_bias_dtype == x_dtype) {
        PADDLE_LAUNCH_LAYERNORM_BWD(T, true);
      } else {
        PADDLE_LAUNCH_LAYERNORM_BWD(U, false);
      }
  }

#undef PADDLE_LAUNCH_LAYERNORM_BWD
#undef PADDLE_LAUNCH_FAST_LAYERNORM_V2_BWD
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(layer_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormGradKernel,
                   float,
                   phi::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#elif CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(layer_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#else
PD_REGISTER_KERNEL(layer_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormGradKernel,
                   float,
                   double,
                   phi::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif
