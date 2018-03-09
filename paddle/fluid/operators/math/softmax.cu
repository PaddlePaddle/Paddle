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
void SoftmaxCUDNNFunctor<T>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor* X,
    framework::Tensor* Y) {
  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor xDesc;
  ScopedTensorDescriptor yDesc;
  DataLayout layout = DataLayout::kNCHW;

  cudnnTensorDescriptor_t cudnn_x_desc =
      xDesc.descriptor<T>(layout, framework::vectorize2int(X->dims()));
  cudnnTensorDescriptor_t cudnn_y_desc =
      xDesc.descriptor<T>(layout, framework::vectorize2int(Y->dims()));
  // NOTE(*) The signature of cudnnSoftmaxForward
  // final = alpha[0]*softmax + beta[0]*priorDstValue.
  Tensor alpha, beta;
  alpha.mutable_data<T>(X->dims(), context.GetPlace());
  beta.mutable_data<T>(X->dims(), context.GetPlace());
  // alpha.Resize(X->dims());
  // beta.Resize(X->dims());
  math::SetConstant<platform::CUDADeviceContext, T> constant;
  constant(context, &alpha, static_cast<T>(1));
  constant(context, &beta, static_cast<T>(0));

  PADDLE_ENFORCE(platform::dynload::cudnnSoftmaxForward(
      context.cudnn_handle(), CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE,
      alpha.data<T>(), cudnn_x_desc, X->data<T>(), beta.data<T>(), cudnn_y_desc,
      Y->mutable_data<T>(context.GetPlace())));
}

template class SoftmaxFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxFunctor<platform::CUDADeviceContext, double>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, float>;
template class SoftmaxGradFunctor<platform::CUDADeviceContext, double>;

template class SoftmaxCUDNNFunctor<float>;
template class SoftmaxCUDNNFunctor<double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
