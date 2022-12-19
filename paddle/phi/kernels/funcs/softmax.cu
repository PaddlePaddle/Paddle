/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"

namespace phi {
namespace funcs {

using ScopedTensorDescriptor = phi::backends::gpu::ScopedTensorDescriptor;
using DataLayout = phi::backends::gpu::DataLayout;
template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;

template <typename T, typename DeviceContext>
void SoftmaxCUDNNFunctor<T, DeviceContext>::operator()(
    const DeviceContext& context,
    const phi::DenseTensor* X,
    phi::DenseTensor* Y) {
  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor xDesc;
  ScopedTensorDescriptor yDesc;
  std::vector<int> cudnn_tensor_dims = phi::vectorize<int>(X->dims());
  DataLayout layout = DataLayout::kNCHW;
  if (cudnn_tensor_dims.size() == 5) {
    layout = DataLayout::kNCDHW;
  }
  // NOTE(*) : cudnn softmax only support >= 4D phi::DenseTensor,
  // fill 1 at unused dims
  if (cudnn_tensor_dims.size() <= 2) {
    cudnn_tensor_dims.resize(4, 1);
  }
#ifdef PADDLE_WITH_HIP
  miopenTensorDescriptor_t cudnn_x_desc =
      xDesc.descriptor<T>(layout, cudnn_tensor_dims);
  miopenTensorDescriptor_t cudnn_y_desc =
      xDesc.descriptor<T>(layout, cudnn_tensor_dims);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenSoftmaxForward_V2(context.cudnn_handle(),
                                            CudnnDataType<T>::kOne(),
                                            cudnn_x_desc,
                                            X->data<T>(),
                                            CudnnDataType<T>::kZero(),
                                            cudnn_y_desc,
                                            context.template Alloc<T>(Y),
                                            MIOPEN_SOFTMAX_ACCURATE,
                                            MIOPEN_SOFTMAX_MODE_INSTANCE));
#else
  cudnnTensorDescriptor_t cudnn_x_desc =
      xDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_y_desc =
      xDesc.descriptor<T>(layout, cudnn_tensor_dims);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnSoftmaxForward(context.cudnn_handle(),
                                        CUDNN_SOFTMAX_ACCURATE,
                                        CUDNN_SOFTMAX_MODE_INSTANCE,
                                        CudnnDataType<T>::kOne(),
                                        cudnn_x_desc,
                                        X->data<T>(),
                                        CudnnDataType<T>::kZero(),
                                        cudnn_y_desc,
                                        context.template Alloc<T>(Y)));
#endif
}

template <typename T, typename DeviceContext>
void SoftmaxGradCUDNNFunctor<T, DeviceContext>::operator()(
    const DeviceContext& context,
    const phi::DenseTensor* Y,
    const phi::DenseTensor* YGrad,
    phi::DenseTensor* XGrad) {
  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor yDesc;
  ScopedTensorDescriptor dyDesc;
  ScopedTensorDescriptor dxDesc;
  std::vector<int> cudnn_tensor_dims = phi::vectorize<int>(Y->dims());
  DataLayout layout = DataLayout::kNCHW;
  if (cudnn_tensor_dims.size() == 5) {
    layout = DataLayout::kNCDHW;
  }
  // NOTE(*) : cudnn softmax only support >= 4D phi::DenseTensor,
  // fill 1 at unused dims
  if (cudnn_tensor_dims.size() <= 2) {
    cudnn_tensor_dims.resize(4, 1);
  }
#ifdef PADDLE_WITH_HIP
  miopenTensorDescriptor_t cudnn_y_desc =
      yDesc.descriptor<T>(layout, cudnn_tensor_dims);
  miopenTensorDescriptor_t cudnn_xgrad_desc =
      dxDesc.descriptor<T>(layout, cudnn_tensor_dims);
  miopenTensorDescriptor_t cudnn_ygrad_desc =
      dyDesc.descriptor<T>(layout, cudnn_tensor_dims);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::miopenSoftmaxBackward_V2(context.cudnn_handle(),
                                             CudnnDataType<T>::kOne(),
                                             cudnn_y_desc,
                                             Y->data<T>(),
                                             cudnn_ygrad_desc,
                                             YGrad->data<T>(),
                                             CudnnDataType<T>::kZero(),
                                             cudnn_xgrad_desc,
                                             context.template Alloc<T>(XGrad),
                                             MIOPEN_SOFTMAX_ACCURATE,
                                             MIOPEN_SOFTMAX_MODE_INSTANCE));
#else
  cudnnTensorDescriptor_t cudnn_y_desc =
      yDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_xgrad_desc =
      dxDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_ygrad_desc =
      dyDesc.descriptor<T>(layout, cudnn_tensor_dims);
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnSoftmaxBackward(context.cudnn_handle(),
                                         CUDNN_SOFTMAX_ACCURATE,
                                         CUDNN_SOFTMAX_MODE_INSTANCE,
                                         CudnnDataType<T>::kOne(),
                                         cudnn_y_desc,
                                         Y->data<T>(),
                                         cudnn_ygrad_desc,
                                         YGrad->data<T>(),
                                         CudnnDataType<T>::kZero(),
                                         cudnn_xgrad_desc,
                                         context.template Alloc<T>(XGrad)));
#endif
}

template class SoftmaxCUDNNFunctor<float, phi::GPUContext>;
template class SoftmaxCUDNNFunctor<phi::dtype::float16, phi::GPUContext>;
template class SoftmaxGradCUDNNFunctor<float, phi::GPUContext>;
template class SoftmaxGradCUDNNFunctor<phi::dtype::float16, phi::GPUContext>;
#if CUDNN_VERSION_MIN(8, 1, 0)
template class SoftmaxCUDNNFunctor<phi::dtype::bfloat16, phi::GPUContext>;
template class SoftmaxGradCUDNNFunctor<phi::dtype::bfloat16, phi::GPUContext>;
#endif

// MIOPEN do not support double
#ifndef PADDLE_WITH_HIP
template class SoftmaxCUDNNFunctor<double, phi::GPUContext>;
template class SoftmaxGradCUDNNFunctor<double, phi::GPUContext>;
#endif

template class SoftmaxFunctor<phi::GPUContext, phi::dtype::float16>;
template class SoftmaxFunctor<phi::GPUContext, phi::dtype::bfloat16>;
template class SoftmaxFunctor<phi::GPUContext, float>;
template class SoftmaxFunctor<phi::GPUContext, double>;
template class SoftmaxGradFunctor<phi::GPUContext, float>;
template class SoftmaxGradFunctor<phi::GPUContext, double>;
template class SoftmaxGradFunctor<phi::GPUContext, phi::dtype::float16>;
template class SoftmaxGradFunctor<phi::GPUContext, phi::dtype::bfloat16>;

}  // namespace funcs
}  // namespace phi
