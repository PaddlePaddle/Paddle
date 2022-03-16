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

#include "paddle/phi/kernels/layer_norm_kernel.h"

#include "paddle/fluid/operators/layer_norm_kernel.cu.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/layer_norm_util.h"

namespace phi {

template <typename T>
void LayerNormDirectCUDAFunctor<T>::operator()(gpuStream_t stream,
                                               const T *input,
                                               std::vector<int> input_shape,
                                               const T *bias,
                                               const T *scale,
                                               T *output,
                                               T *mean,
                                               T *variance,
                                               int begin_norm_axis,
                                               float eps) {
  const auto x_dims = phi::make_ddim(input_shape);
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);
  switch (paddle::operators::GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(paddle::operators::LayerNormForward<
                         T,
                         T,
                         kBlockDim><<<batch_size, kBlockDim, 0, stream>>>(
        input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
}

template class LayerNormDirectCUDAFunctor<float>;

template <typename T, typename Context>
void LayerNormKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     paddle::optional<const DenseTensor &> scale_opt,
                     paddle::optional<const DenseTensor &> bias_opt,
                     float epsilon,
                     int begin_norm_axis,
                     bool is_test,
                     DenseTensor *y,
                     DenseTensor *mean,
                     DenseTensor *var) {
  using U = paddle::operators::LayerNormParamType<T>;
  auto *scale = scale_opt.get_ptr();
  auto *bias = bias_opt.get_ptr();

  const auto x_dims = x.dims();
  auto *x_data = x.data<T>();
  auto *y_data = dev_ctx.template Alloc<T>(y);
  auto *mean_data = dev_ctx.template Alloc<U>(mean);
  auto *var_data = dev_ctx.template Alloc<U>(var);

  auto *void_scale_data = (scale == nullptr ? nullptr : scale->data());
  auto *void_bias_data = (bias == nullptr ? nullptr : bias->data());

  auto x_dtype = x.dtype();
  phi::DataType scale_bias_dtype;
  if (void_scale_data != nullptr) {
    scale_bias_dtype = scale->dtype();
    if (void_bias_data != nullptr) {
      PADDLE_ENFORCE_EQ(
          scale->dtype(),
          bias->dtype(),
          phi::errors::InvalidArgument("Thie Scale and Bias of layer_norm op "
                                       "should have the same data type."));
    }
  } else {
    scale_bias_dtype = (void_bias_data != nullptr ? bias->dtype() : x_dtype);
  }

  bool is_scale_bias_same_dtype_with_x = x_dtype == scale_bias_dtype;
  if (!is_scale_bias_same_dtype_with_x) {
    PADDLE_ENFORCE_EQ(scale_bias_dtype,
                      paddle::experimental::CppTypeToDataType<U>::Type(),
                      phi::errors::InvalidArgument(
                          "Unsupported data type of Scale and Bias"));
  }

  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);

  auto stream = dev_ctx.stream();

#define PADDLE_LAUNCH_LAYERNORM_FWD(ScaleBiasT, IsScaleBiasSameDTypeWithX) \
  do {                                                                     \
    switch (paddle::operators::GetDesiredBlockDim(feature_size)) {         \
      FIXED_BLOCK_DIM_CASE(paddle::operators::LayerNormForward<            \
                           T,                                              \
                           U,                                              \
                           kBlockDim,                                      \
                           IsScaleBiasSameDTypeWithX><<<batch_size,        \
                                                        kBlockDim,         \
                                                        0,                 \
                                                        stream>>>(         \
          x_data,                                                          \
          static_cast<const ScaleBiasT *>(void_scale_data),                \
          static_cast<const ScaleBiasT *>(void_bias_data),                 \
          y_data,                                                          \
          mean_data,                                                       \
          var_data,                                                        \
          epsilon,                                                         \
          feature_size));                                                  \
      default:                                                             \
        PADDLE_THROW(phi::errors::InvalidArgument(                         \
            "Product from begin_norm_axis to end must be larger than 1")); \
        break;                                                             \
    }                                                                      \
  } while (0)

#ifdef PADDLE_WITH_CUDA
  bool can_call_1024_kernel = false;
  if (feature_size == 1024 && scale != nullptr && bias != nullptr) {
    can_call_1024_kernel = true;
  }
  if (can_call_1024_kernel) {
    const int WARPS_M = 4;
    const int WARPS_N = 1;
    const int THREADS_PER_WARP = 32;
    const int BYTES_PER_LDG = 16;
    const int VecSize = BYTES_PER_LDG / sizeof(T);

    const int THREADS_PER_CTA = WARPS_N * THREADS_PER_WARP * WARPS_M;
    const int ROWS_PER_CTA = WARPS_M;

    const int grid = static_cast<int>(
        std::ceil(batch_size / static_cast<float>(ROWS_PER_CTA)));
    if (is_scale_bias_same_dtype_with_x) {
      paddle::operators::ln_fwd_1024_kernel<
          T,
          U,
          T,
          VecSize,
          WARPS_M,
          WARPS_N,
          BYTES_PER_LDG><<<grid, THREADS_PER_CTA, 0, stream>>>(
          batch_size,
          feature_size,
          epsilon,
          x_data,
          static_cast<const T *>(void_scale_data),
          static_cast<const T *>(void_bias_data),
          mean_data,
          var_data,
          y_data);
    } else {
      paddle::operators::ln_fwd_1024_kernel<
          T,
          U,
          U,
          VecSize,
          WARPS_M,
          WARPS_N,
          BYTES_PER_LDG><<<grid, THREADS_PER_CTA, 0, stream>>>(
          batch_size,
          feature_size,
          epsilon,
          x_data,
          static_cast<const U *>(void_scale_data),
          static_cast<const U *>(void_bias_data),
          mean_data,
          var_data,
          y_data);
    }
  } else {
#endif
    if (is_scale_bias_same_dtype_with_x) {
      PADDLE_LAUNCH_LAYERNORM_FWD(T, true);
    } else {
      PADDLE_LAUNCH_LAYERNORM_FWD(U, false);
    }
#ifdef PADDLE_WITH_CUDA
  }
#endif

#undef PADDLE_LAUNCH_LAYERNORM_FWD
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   phi::dtype::float16) {}
#elif CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
