/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ReluMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // auto* x = ctx.Input<Tensor>("X");
    // auto* out = ctx.Output<Tensor>("Out");

    // out->mutable_data<T>(ctx.GetPlace());

    // auto stream =
    //     ctx.template device_context<paddle::platform::MLUDeviceContext>().stream();
    // TODO(fwg): impl cnnl relu
    LOG(INFO) << "MLU relu kernel.";
  }
};

template <typename DeviceContext, typename T>
class ReluGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // auto* out = ctx.Input<Tensor>("Out");
    // auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    // auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    // auto stream =
    //     ctx.template device_context<paddle::platform::MLUDeviceContext>().stream();

    // dx->mutable_data<T>(ctx.GetPlace());
    // TODO(fwg): impl cnnl relugrad
    LOG(INFO) << "MLU relugrad kernel.";

  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    relu,
    ops::ReluMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::ReluMLUKernel<paddle::platform::MLUDeviceContext, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    relu_grad,
    ops::ReluGradMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::ReluGradMLUKernel<paddle::platform::MLUDeviceContext, paddle::platform::float16>);