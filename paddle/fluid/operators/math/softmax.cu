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

#define EIGEN_USE_GPU

#include <vector>

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
void SoftmaxCUDNNFunctor<T>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor* X,
    framework::Tensor* Y) {
  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor xDesc;
  ScopedTensorDescriptor yDesc;
  std::vector<int> cudnn_tensor_dims = framework::vectorize2int(X->dims());
  DataLayout layout = DataLayout::kNCHW;
  if (cudnn_tensor_dims.size() == 5) {
    layout = DataLayout::kNCDHW;
  }
  // NOTE(*) : cudnn softmax only support >= 4D Tensor,
  // fill 1 at unused dims
  if (cudnn_tensor_dims.size() <= 2) {
    cudnn_tensor_dims.resize(4, 1);
  }
  cudnnTensorDescriptor_t cudnn_x_desc =
      xDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_y_desc =
      xDesc.descriptor<T>(layout, cudnn_tensor_dims);
  PADDLE_ENFORCE(platform::dynload::cudnnSoftmaxForward(
      context.cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_INSTANCE, CudnnDataType<T>::kOne(), cudnn_x_desc,
      X->data<T>(), CudnnDataType<T>::kZero(), cudnn_y_desc,
      Y->mutable_data<T>(context.GetPlace())));
}

template <typename T>
void SoftmaxGradCUDNNFunctor<T>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor* Y,
    const framework::Tensor* YGrad, framework::Tensor* XGrad) {
  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor yDesc;
  ScopedTensorDescriptor dyDesc;
  ScopedTensorDescriptor dxDesc;
  std::vector<int> cudnn_tensor_dims = framework::vectorize2int(Y->dims());
  DataLayout layout = DataLayout::kNCHW;
  if (cudnn_tensor_dims.size() == 5) {
    layout = DataLayout::kNCDHW;
  }
  // NOTE(*) : cudnn softmax only support >= 4D Tensor,
  // fill 1 at unused dims
  if (cudnn_tensor_dims.size() <= 2) {
    cudnn_tensor_dims.resize(4, 1);
  }
  cudnnTensorDescriptor_t cudnn_y_desc =
      yDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_xgrad_desc =
      dxDesc.descriptor<T>(layout, cudnn_tensor_dims);
  cudnnTensorDescriptor_t cudnn_ygrad_desc =
      dyDesc.descriptor<T>(layout, cudnn_tensor_dims);
  PADDLE_ENFORCE(platform::dynload::cudnnSoftmaxBackward(
      context.cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_INSTANCE, CudnnDataType<T>::kOne(), cudnn_y_desc,
      Y->data<T>(), cudnn_ygrad_desc, YGrad->data<T>(),
      CudnnDataType<T>::kZero(), cudnn_xgrad_desc,
      XGrad->mutable_data<T>(context.GetPlace())));
}

template class SoftmaxCUDNNFunctor<platform::float16>;
template class SoftmaxCUDNNFunctor<float>;
template class SoftmaxCUDNNFunctor<double>;
template class SoftmaxGradCUDNNFunctor<float>;
template class SoftmaxGradCUDNNFunctor<double>;

template class SoftmaxFunctor<platform::CUDADeviceContext, platform::float16>;
template class SoftmaxFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxFunctor<platform::CUDADeviceContext, double>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
