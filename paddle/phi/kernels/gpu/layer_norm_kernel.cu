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

template <typename T, typename U>
void LayerNormDirectCUDAFunctor<T, U>::operator()(gpuStream_t stream,
                                                  const T *input,
                                                  std::vector<int> input_shape,
                                                  const U *bias,
                                                  const U *scale,
                                                  T *output,
                                                  U *mean,
                                                  U *variance,
                                                  int begin_norm_axis,
                                                  float eps) {
  const auto x_dims = phi::make_ddim(input_shape);
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);
  switch (paddle::operators::GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        paddle::operators::LayerNormForward<T, U, kBlockDim>
        <<<batch_size, kBlockDim, 0, stream>>>(
            input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
}

// #if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
template <>
void LayerNormDirectCUDAFunctor<half, float>::operator()(
    gpuStream_t stream,
    const half *input,
    std::vector<int> input_shape,
    const float *bias,
    const float *scale,
    half *output,
    float *mean,
    float *variance,
    int begin_norm_axis,
    float eps) {
  const auto x_dims = phi::make_ddim(input_shape);
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);
  switch (paddle::operators::GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        paddle::operators::LayerNormForwardFP16<kBlockDim>
        <<<batch_size, kBlockDim, 0, stream>>>(
            input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
}
// #endif

template class LayerNormDirectCUDAFunctor<float, float>;
template class LayerNormDirectCUDAFunctor<half, float>;

template <typename T, typename Context>
void LayerNormKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     const paddle::optional<DenseTensor> &scale_opt,
                     const paddle::optional<DenseTensor> &bias_opt,
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
          phi::errors::InvalidArgument("This Scale and Bias of layer_norm op "
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
      FIXED_BLOCK_DIM_CASE(                                                \
          paddle::operators::                                              \
              LayerNormForward<T, U, kBlockDim, IsScaleBiasSameDTypeWithX> \
          <<<batch_size, kBlockDim, 0, stream>>>(                          \
              x_data,                                                      \
              static_cast<const ScaleBiasT *>(void_scale_data),            \
              static_cast<const ScaleBiasT *>(void_bias_data),             \
              y_data,                                                      \
              mean_data,                                                   \
              var_data,                                                    \
              epsilon,                                                     \
              feature_size));                                              \
      default:                                                             \
        PADDLE_THROW(phi::errors::InvalidArgument(                         \
            "Product from begin_norm_axis to end must be larger than 1")); \
        break;                                                             \
    }                                                                      \
  } while (0)

#define PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, feature_size)          \
  case (feature_size): {                                                     \
    constexpr int WARPS_N = feature_size < 1024 ? 1 : (feature_size / 1024); \
    constexpr int WARPS_M = 4 / WARPS_N;                                     \
    const int THREADS_PER_WARP = 32;                                         \
    const int BYTES_PER_LDG = 16;                                            \
    const int VecSize = BYTES_PER_LDG / sizeof(T);                           \
    const int THREADS_PER_CTA = WARPS_N * THREADS_PER_WARP * WARPS_M;        \
    const int ROWS_PER_CTA = WARPS_M;                                        \
    const int grid = static_cast<int>(                                       \
        std::ceil(batch_size / static_cast<float>(ROWS_PER_CTA)));           \
    paddle::operators::fast_ln_fwd_kernel<T,                                 \
                                          U,                                 \
                                          ScaleT,                            \
                                          VecSize,                           \
                                          WARPS_M,                           \
                                          WARPS_N,                           \
                                          BYTES_PER_LDG>                     \
        <<<grid, THREADS_PER_CTA, 0, stream>>>(                              \
            batch_size,                                                      \
            feature_size,                                                    \
            epsilon,                                                         \
            x_data,                                                          \
            static_cast<const ScaleT *>(void_scale_data),                    \
            static_cast<const ScaleT *>(void_bias_data),                     \
            mean_data,                                                       \
            var_data,                                                        \
            y_data);                                                         \
  } break

#define PADDLE_LAUNCH_FAST_LAYERNORM_FWD(ScaleT)       \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 768);  \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 1024); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 1280); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 1536); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 1792); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 2048); \
  PADDLE_LAUNCH_FAST_LAYERNORM_FWD_BASE(ScaleT, 4096)

#ifdef PADDLE_WITH_CUDA
  bool can_call_fast_kernel = false;
  if ((feature_size >= 768 && feature_size <= 2048 && feature_size % 256 == 0 ||
       feature_size == 4096) &&
      scale != nullptr && bias != nullptr) {
    // can_call_fast_kernel = true;
    can_call_fast_kernel = false;
  }

  if (can_call_fast_kernel) {
    if (is_scale_bias_same_dtype_with_x) {
      switch (feature_size) {
        PADDLE_LAUNCH_FAST_LAYERNORM_FWD(T);
        default:
          PADDLE_THROW(phi::errors::InvalidArgument(
              "Only when feature_size is from 256 to 4096 and is diviaible by "
              "256 is supported "
              "now"));
          break;
      }
    } else {
      switch (feature_size) {
        PADDLE_LAUNCH_FAST_LAYERNORM_FWD(U);
        default:
          PADDLE_THROW(phi::errors::InvalidArgument(
              "Only when feature_size is from 256 to 4096 and is diviaible by "
              "is supported "
              "now"));
          break;
      }
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
#undef PADDLE_LAUNCH_FAST_LAYERNORM_FWD
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
