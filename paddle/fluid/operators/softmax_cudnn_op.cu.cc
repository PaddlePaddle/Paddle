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

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());

    auto dims = X->dims();
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);
    framework::LoDTensor flattened_x;
    framework::LoDTensor flattened_out;
    flattened_x.ShareDataWith(*X).Resize(flattened_dims);
    flattened_out.ShareDataWith(*Out).Resize(flattened_dims);

    math::SoftmaxCUDNNFunctor<T>()(
        context.template device_context<platform::CUDADeviceContext>(),
        &flattened_x, &flattened_out);
  }
};

template <typename T>
class SoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* Out = context.Input<Tensor>("Out");
    auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));

    // allocate memory on device.
    dX->mutable_data<T>(context.GetPlace());

    auto dims = Out->dims();
    auto flattened_dims = framework::flatten_to_2d(dims, dims.size() - 1);
    framework::LoDTensor flattened_out;
    framework::LoDTensor flattened_d_out;
    framework::LoDTensor flattened_d_x;
    flattened_out.ShareDataWith(*Out).Resize(flattened_dims);
    flattened_d_out.ShareDataWith(*dOut).Resize(flattened_dims);
    flattened_d_x.ShareDataWith(*dX).Resize(flattened_dims);

    math::SoftmaxGradCUDNNFunctor<T>()(
        context.template device_context<platform::CUDADeviceContext>(),
        &flattened_out, &flattened_d_out, &flattened_d_x);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
