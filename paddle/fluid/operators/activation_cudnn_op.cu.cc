/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

template <typename T, cudnnActivationMode_t Mode>
class ActivationCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");

    platform::ScopedActivationDescriptor activation_desc;
    platform::ScopedTensorDescriptor x_desc;
    std::vector<int> cudnn_tensor_dims = framework::vectorize2int(x->dims());
    platform::DataLayout layout = platform::DataLayout::kNCHW;
    if (cudnn_tensor_dims.size() == 5) {
      layout = platform::DataLayout::kNCDHW;
    }

    cudnnTensorDescriptor_t cudnn_x_desc =
        x_desc.descriptor<T>(layout, cudnn_tensor_dims);

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::dynload::cudnnActivationForward(
        dev_ctx.cudnn_handle(), activation_desc.descriptor(Mode),
        platform::CudnnDataType<T>::kOne(), cudnn_x_desc, x->data<T>(),
        platform::CudnnDataType<T>::kZero(), cudnn_x_desc,
        out->mutable_data<T>(context.GetPlace())));
  }
};

template <typename T, cudnnActivationMode_t Mode>
class ActivationGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));

    platform::ScopedActivationDescriptor activation_desc;
    platform::ScopedTensorDescriptor x_desc;
    std::vector<int> cudnn_tensor_dims = framework::vectorize2int(x->dims());
    platform::DataLayout layout = platform::DataLayout::kNCHW;
    if (cudnn_tensor_dims.size() == 5) {
      layout = platform::DataLayout::kNCDHW;
    }

    cudnnTensorDescriptor_t cudnn_x_desc =
        x_desc.descriptor<T>(layout, cudnn_tensor_dims);

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    PADDLE_ENFORCE(platform::dynload::cudnnActivationBackward(
        dev_ctx.cudnn_handle(), activation_desc.descriptor(Mode),
        platform::CudnnDataType<T>::kOne(), cudnn_x_desc, out->data<T>(),
        cudnn_x_desc, out_grad->data<T>(), cudnn_x_desc, x->data<T>(),
        platform::CudnnDataType<T>::kZero(), cudnn_x_desc,
        x_grad->mutable_data<T>(context.GetPlace())));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(
    relu, CUDNN, ::paddle::platform::CUDAPlace,
    paddle::operators::ActivationCUDNNKernel<float, CUDNN_ACTIVATION_RELU>,
    paddle::operators::ActivationCUDNNKernel<double, CUDNN_ACTIVATION_RELU>);
REGISTER_OP_KERNEL(
    relu_grad, CUDNN, ::paddle::platform::CUDAPlace,
    paddle::operators::ActivationGradCUDNNKernel<float, CUDNN_ACTIVATION_RELU>,
    paddle::operators::ActivationGradCUDNNKernel<double,
                                                 CUDNN_ACTIVATION_RELU>);
