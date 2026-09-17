/* Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/phi/kernels/funcs/fc_functor.h"

#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/fc_functor_gpu.h"
#include "paddle/phi/kernels/matmul_kernel.h"

namespace phi {
namespace funcs {

template <typename DeviceContext, typename T>
void FCFunctor<DeviceContext, T>::operator()(const DeviceContext& dev_ctx,
                                             const int M,
                                             const int N,
                                             const int K,
                                             const T* X,
                                             const T* W,
                                             T* Y,
                                             const T* B,
                                             bool relu,
                                             bool padding_weights) {
  PADDLE_ENFORCE_EQ(padding_weights,
                    false,
                    errors::PermissionDenied(
                        "Weight padding in fc can not be used in GPU scope."));
  auto blas = funcs::GetBlas<DeviceContext, T>(dev_ctx);
  blas.GEMM(CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            K,
            static_cast<T>(1.0),
            X,
            W,
            static_cast<T>(0.0),
            Y);
  if (B == nullptr) {
    return;
  }

  AddReluKernel(dev_ctx.stream(), M, N, Y, B, relu);
}

template class FCFunctor<GPUContext, float16>;
template class FCFunctor<GPUContext, float>;
template class FCFunctor<GPUContext, double>;

template <typename DeviceContext, typename T>
void FCInt8Functor<DeviceContext, T>::operator()(
    const DeviceContext& dev_ctx,
    const int M,
    const int N,
    const int K,
    const T* X,
    const DenseTensor* w_tensor,
    T* Y,
    float scale_in,
    std::vector<float> scale_weights,
    int quant_round_type,
    float quant_max_bound,
    float quant_min_bound,
    const T* B,
    bool relu,
    bool padding_weights) {
  PADDLE_ENFORCE_EQ(padding_weights,
                    false,
                    errors::PermissionDenied(
                        "Weight padding in fc can not be used in GPU scope."));

  DenseTensor quant_x_tensor, quant_y_tensor;
  quant_x_tensor.Resize({M, K});
  quant_y_tensor.Resize({M, N});
  dev_ctx.template Alloc<int8_t>(&quant_x_tensor,
                                 quant_x_tensor.numel() * sizeof(int8_t));
  dev_ctx.template Alloc<int32_t>(&quant_y_tensor,
                                  quant_y_tensor.numel() * sizeof(int32_t));
  LaunchFcQuantKernel<T>(X,
                         quant_x_tensor.data<int8_t>(),
                         scale_in,
                         M,
                         K,
                         quant_round_type,
                         quant_max_bound,
                         quant_min_bound,
                         dev_ctx.stream());

  MatmulKernel<int8_t, GPUContext>(
      dev_ctx, quant_x_tensor, *w_tensor, false, false, &quant_y_tensor);

  DenseTensor scale_weights_dev;
  scale_weights_dev.Resize({N});
  dev_ctx.template Alloc<float>(&scale_weights_dev,
                                scale_weights_dev.numel() * sizeof(float));
  float* scale_weights_dev_ptr = scale_weights_dev.data<float>();
#ifdef PADDLE_WITH_HIP
  hipMemcpyAsync(scale_weights_dev_ptr,
                 scale_weights.data(),
                 N * sizeof(float),
                 hipMemcpyHostToDevice);
#else
  cudaMemcpyAsync(scale_weights_dev_ptr,
                  scale_weights.data(),
                  N * sizeof(float),
                  cudaMemcpyHostToDevice);
#endif

  LaunchFcDequantKernel(dev_ctx,
                        quant_y_tensor.data<int32_t>(),
                        Y,
                        M,
                        N,
                        scale_in,
                        scale_weights_dev_ptr,
                        quant_max_bound);

  if (B == nullptr) {
    return;
  }

  AddReluKernel(dev_ctx.stream(), M, N, Y, B, relu);
}

template class FCInt8Functor<GPUContext, float16>;
template class FCInt8Functor<GPUContext, float>;
template class FCInt8Functor<GPUContext, double>;

}  // namespace funcs
}  // namespace phi

#endif
