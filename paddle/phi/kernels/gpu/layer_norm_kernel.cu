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
template<typename T>
__global__ void print_float(const T *src, int64_t start_index, int64_t end_index){
  for (int i=start_index;i<end_index;i++){
    printf("%f ",static_cast<double>(src[i]));
    if(i%49==48){
      printf("\r\n");
    }
  }
}
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
  cudaDeviceSynchronize();
  int inputnum_print=1;
  for(int i=0;i<input_shape.size();i++){
    inputnum_print*=input_shape[i];
  }
  inputnum_print=49;
  cudaDeviceSynchronize();

  printf("@#@@@ LayerNormDirectCUDAFunctor input data \r\n");
  print_float<T><<<1,1>>>(input,0,inputnum_print);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormDirectCUDAFunctor scale data \r\n");
  print_float<T><<<1,1>>>(scale,0,inputnum_print);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormDirectCUDAFunctor bias data \r\n");
  print_float<T><<<1,1>>>(bias,0,inputnum_print);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormDirectCUDAFunctor mean data before calculation \r\n");
  print_float<T><<<1,1>>>(mean,0,5);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormDirectCUDAFunctor variance data before calculation \r\n");
  print_float<T><<<1,1>>>(variance,0,5);
  cudaDeviceSynchronize();
  printf("\r\n");

  printf("@@@ begin_norm_axis: %d \r\n", begin_norm_axis);
  printf("@@@ eps: %f\r\n", eps);
  const auto x_dims = phi::make_ddim(input_shape);
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t batch_size = static_cast<int64_t>(matrix_dim[0]);
  int64_t feature_size = static_cast<int64_t>(matrix_dim[1]);

  printf("@@ batch_size: %d, feature_size: %d \r\n",
        batch_size, feature_size);
        
  switch (paddle::operators::GetDesiredBlockDim(feature_size)) {
    FIXED_BLOCK_DIM_CASE(
        paddle::operators::LayerNormForward<T, T, kBlockDim, true>
        <<<batch_size, kBlockDim, 0, stream>>>(
            input, scale, bias, output, mean, variance, eps, feature_size));
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Product from begin_norm_axis to end in layer_norm must be larger "
          "than 1"));
      break;
  }
  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormDirectCUDAFunctor output data \r\n");
  print_float<T><<<1,1>>>(output,0,inputnum_print);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormDirectCUDAFunctor mean data after calculation \r\n");
  print_float<T><<<1,1>>>(mean,0,5);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormDirectCUDAFunctor variance data after calculation \r\n");
  print_float<T><<<1,1>>>(variance,0,5);
  cudaDeviceSynchronize();
  printf("\r\n");
}

template class LayerNormDirectCUDAFunctor<float>;

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
  //print shape of var
  const auto var_dim=var->dims();
  printf("@@@@ var shape in LayerNormKernel \r\n");
  for(int i=0;i<var_dim.size();i++){
    printf("%d, ", var_dim[i]);
  }
  printf("\r\n");
  using U = paddle::operators::LayerNormParamType<T>;
  auto *scale = scale_opt.get_ptr();
  auto *bias = bias_opt.get_ptr();

  const auto x_dims = x.dims();
  auto *x_data = x.data<T>();

  int inputnum_print=x.numel();
  inputnum_print=49;
  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormKernel input data \r\n");
  print_float<T><<<1,1>>>(x_data,0,inputnum_print);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormKernel scale data \r\n");
  print_float<float><<<1,1>>>(static_cast<const float *>(scale->data()),0,inputnum_print);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormKernel bias data \r\n");
  print_float<float><<<1,1>>>(static_cast<const float *>(bias->data()),0,inputnum_print);
  cudaDeviceSynchronize();
  printf("\r\n");

  auto *y_data = dev_ctx.template Alloc<T>(y);
  auto *mean_data = dev_ctx.template Alloc<U>(mean);
  auto *var_data = dev_ctx.template Alloc<U>(var);

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormKernel mean data before calculation \r\n");
  print_float<U><<<1,1>>>(mean_data,0,5);
  cudaDeviceSynchronize();
  printf("\r\n");

  cudaDeviceSynchronize();
  printf("@#@@@ LayerNormKernel variance data before calculation \r\n");
  print_float<U><<<1,1>>>(var_data,0,5);
  cudaDeviceSynchronize();
  printf("\r\n");

  printf("@@@ begin_norm_axis: %d \r\n", begin_norm_axis);
  printf("@@@ eps: %f\r\n", epsilon);

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
  printf("@@ batch_size: %d, feature_size: %d \r\n",
        batch_size,feature_size);
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

cudaDeviceSynchronize();
printf("@#@@@ LayerNormKernel output data \r\n");
print_float<T><<<1,1>>>(y_data,0,inputnum_print);
cudaDeviceSynchronize();
printf("\r\n");

cudaDeviceSynchronize();
printf("@#@@@ LayerNormKernel mean data after calculation \r\n");
print_float<U><<<1,1>>>(mean_data,0,5);
cudaDeviceSynchronize();
printf("\r\n");

cudaDeviceSynchronize();
printf("@#@@@ LayerNormKernel variance data after calculation \r\n");
print_float<U><<<1,1>>>(var_data,0,5);
cudaDeviceSynchronize();
printf("\r\n");

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
