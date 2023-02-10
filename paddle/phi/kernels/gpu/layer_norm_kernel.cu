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

template <typename U>
__device__ void WelfordOnline(U val, U *mean, U *square, U *count) {
  *count += 1;
  U delta1 = val - *mean;
  *mean += delta1 / (*count);
  U delta2 = val - *mean;
  *square += delta1 * delta2;
}

template <typename U>
__device__ void WelfordOnline(
    U b_mean, U b_square, U b_cnt, U *mean, U *square, U *count) {
  if (b_cnt == 0) {
    return;
  }
  U new_cnt = *count + b_cnt;
  U nb_n = b_cnt / new_cnt;
  U delta = b_mean - *mean;
  *mean += delta * nb_n;
  *square += b_square + delta * delta * (*count) * nb_n;
  *count = new_cnt;
}

template <typename U>
__device__ void WelfordWarpAllReduce(
    U tid_mean, U tid_square, U tid_cnt, U *mean, U *square, U *count) {
  constexpr int kWarpSize = 32;
  *mean = tid_mean;
  *square = tid_square;
  *count = tid_cnt;
  for (int mask = 1; mask < kWarpSize; mask *= 2) {
    U b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
    U b_square = __shfl_down_sync(0xffffffff, *square, mask);
    U b_cnt = __shfl_down_sync(0xffffffff, *count, mask);
    WelfordOnline<U>(b_mean, b_square, b_cnt, mean, square, count);
  }

  *mean = __shfl_sync(0xffffffff, *mean, 0, kWarpSize);
  *square = __shfl_sync(0xffffffff, *square, 0, kWarpSize);
  *count = __shfl_sync(0xffffffff, *count, 0, kWarpSize);
}

template <typename T, typename U, bool IsSameType>
__global__ void LayerNormFwdWithWelford(
    const T *__restrict__ src_data,
    T *dst_data,
    const paddle::operators::LayerNormScaleBiasT<T, U, IsSameType> *scale,
    const paddle::operators::LayerNormScaleBiasT<T, U, IsSameType> *bias,
    U *mean,
    U *var,
    const float epsilon,
    const int64_t rows,
    const int64_t cols,
    const bool valid_scale,
    const bool valid_bias) {
  constexpr int kWarpSize = 32;
  int row_offset = blockIdx.x * blockDim.y + threadIdx.y;
  int cols_per_thread = (cols + (kWarpSize - 1)) / kWarpSize;
  int cols_this_thread = cols_per_thread;
  int last_y = (cols / cols_per_thread);
  if (threadIdx.x == last_y) {
    cols_this_thread = cols - cols_per_thread * last_y;
  } else if (threadIdx.x > last_y) {
    cols_this_thread = 0;
  }

  if (row_offset < rows) {
    U buffer[32];
    U tid_mean = static_cast<U>(0);
    U tid_cnt = static_cast<U>(0);
    U tid_square = static_cast<U>(0);

    U warp_mean;
    U warp_square;
    U warp_cnt;
    const T *__restrict__ row_input = src_data + row_offset * cols;
    T *row_output = dst_data + row_offset * cols;

    for (int i = 0; i < cols_this_thread; i++) {
      buffer[i] = static_cast<U>(row_input[threadIdx.x * cols_per_thread + i]);
    }

    for (int i = 0; i < cols_this_thread; i++) {
      WelfordOnline<U>(buffer[i], &tid_mean, &tid_square, &tid_cnt);
    }
    WelfordWarpAllReduce<U>(
        tid_mean, tid_square, tid_cnt, &warp_mean, &warp_square, &warp_cnt);
    U row_mean = warp_mean;
    U row_variance = max(warp_square / warp_cnt, 0.f);
    U row_inv_var = rsqrt(row_variance + epsilon);

    if (threadIdx.x == 0) {
      mean[row_offset] = row_mean;
      var[row_offset] = row_variance;
    }

    // No scale and no bias
    if ((!valid_scale) && (!valid_bias)) {
#pragma unroll
      for (int i = 0; i < cols_this_thread; ++i) {
        int idx = threadIdx.x * cols_per_thread + i;
        row_output[idx] = static_cast<T>((buffer[i] - row_mean) * row_inv_var);
      }
    } else if (valid_scale && valid_bias) {
      // Has scale and bias
#pragma unroll
      for (int i = 0; i < cols_this_thread; ++i) {
        int idx = threadIdx.x * cols_per_thread + i;
        buffer[i] = (buffer[i] - row_mean) * row_inv_var;
        row_output[idx] = static_cast<T>(
            buffer[i] * static_cast<U>(scale[idx]) + static_cast<U>(bias[idx]));
      }
    } else {
#pragma unroll
      for (int i = 0; i < cols_this_thread; ++i) {
        buffer[i] = (buffer[i] - row_mean) * row_inv_var;
      }
      // Only has scale
      if (valid_scale) {
#pragma unroll
        for (int i = 0; i < cols_this_thread; ++i) {
          int idx = threadIdx.x * cols_per_thread + i;
          row_output[idx] =
              static_cast<T>(buffer[i] * static_cast<U>(scale[idx]));
        }
      } else {
        // Only has bias
#pragma unroll
        for (int i = 0; i < cols_this_thread; ++i) {
          int idx = threadIdx.x * cols_per_thread + i;
          row_output[idx] =
              static_cast<T>(buffer[i] + static_cast<U>(bias[idx]));
        }
      }
    }
  }
}

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

template class LayerNormDirectCUDAFunctor<float, float>;
template class LayerNormDirectCUDAFunctor<double, double>;
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
template class LayerNormDirectCUDAFunctor<half, float>;
#endif

template <typename T, typename Context>
void LayerNormKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     const paddle::optional<DenseTensor> &scale_opt,
                     const paddle::optional<DenseTensor> &bias_opt,
                     float epsilon,
                     int begin_norm_axis,
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

  bool valid_scale = (scale != nullptr);
  bool valid_bias = (bias != nullptr);
  auto *void_scale_data = valid_scale ? scale->data() : nullptr;
  auto *void_bias_data = valid_bias ? bias->data() : nullptr;

  auto x_dtype = x.dtype();
  phi::DataType scale_bias_dtype;
  if (valid_scale) {
    scale_bias_dtype = scale->dtype();
    if (valid_bias) {
      PADDLE_ENFORCE_EQ(
          scale->dtype(),
          bias->dtype(),
          phi::errors::InvalidArgument("This Scale and Bias of layer_norm op "
                                       "should have the same data type."));
    }
  } else {
    scale_bias_dtype = valid_bias ? bias->dtype() : x_dtype;
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

    if (feature_size <= 1024 && (!std::is_same<T, int8_t>::value)) {
      constexpr int warp_size = 32;
      constexpr int row_per_block = 8;
      int64_t block_size = (batch_size + (row_per_block - 1)) / row_per_block;
      dim3 threads(warp_size, row_per_block, 1);

      if (is_scale_bias_same_dtype_with_x) {
        LayerNormFwdWithWelford<T, U, true><<<block_size, threads, 0, stream>>>(
            x_data,
            y_data,
            static_cast<const T *>(void_scale_data),
            static_cast<const T *>(void_bias_data),
            mean_data,
            var_data,
            epsilon,
            batch_size,
            feature_size,
            valid_scale,
            valid_bias);
      } else {
        LayerNormFwdWithWelford<T, U, false>
            <<<block_size, threads, 0, stream>>>(
                x_data,
                y_data,
                static_cast<const U *>(void_scale_data),
                static_cast<const U *>(void_bias_data),
                mean_data,
                var_data,
                epsilon,
                batch_size,
                feature_size,
                valid_scale,
                valid_bias);
      }
    } else {
      if (is_scale_bias_same_dtype_with_x) {
        PADDLE_LAUNCH_LAYERNORM_FWD(T, true);
      } else {
        PADDLE_LAUNCH_LAYERNORM_FWD(U, false);
      }
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
