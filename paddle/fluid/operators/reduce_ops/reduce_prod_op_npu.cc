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

#include "paddle/fluid/operators/reduce_ops/reduce_prod_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class ReduceProdNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(WARNING) << "ReduceProdNPUKernel";

    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    auto dims = ctx.Attr<std::vector<int>>("dim");
    bool keep_dim = ctx.Attr<bool>("keep_dim");

    LOG(WARNING) << "x: " << x;
    LOG(WARNING) << "x numel: " << x->numel();
    LOG(WARNING) << "x dims: " << x->dims();

    LOG(WARNING) << "out: " << out;
    LOG(WARNING) << "out numel: " << out->numel();
    LOG(WARNING) << "out dims: " << out->dims();

    // LOG(WARNING) << "dims: " << dims;
    LOG(WARNING) << "keep_dim: " << keep_dim;

    out->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {{"axes", dims},
                                             {"keep_dims", keep_dim}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("ReduceProdD", {*x}, {*out}, attr_input);
    runner.Run(stream);

    LOG(WARNING) << "out: " << out;
    LOG(WARNING) << "out numel: " << out->numel();
    LOG(WARNING) << "out dims: " << out->dims();
  }
};

template <typename DeviceContext, typename T>
class ReduceProdGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(WARNING) << "ReduceProdGradNPUKernel";

    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto dims = ctx.Attr<std::vector<int>>("dim");
    bool keep_dim = ctx.Attr<bool>("keep_dim");

    LOG(WARNING) << "x: " << x;
    LOG(WARNING) << "x numel: " << x->numel();
    LOG(WARNING) << "x dims: " << x->dims();

    LOG(WARNING) << "dout: " << dout;
    LOG(WARNING) << "dout numel: " << dout->numel();
    LOG(WARNING) << "dout dims: " << dout->dims();

    LOG(WARNING) << "dx: " << dx;
    LOG(WARNING) << "dx numel: " << dx->numel();
    LOG(WARNING) << "dx dims: " << dx->dims();

    // LOG(WARNING) << "dims: " << dims;
    LOG(WARNING) << "keep_dim: " << keep_dim;

    dx->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {{"axes", dims},
                                             {"keep_dims", keep_dim}};

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner_dx =
        NpuOpRunner("ReduceProdD", {*x, *dout}, {*dx}, attr_input);
    runner_dx.Run(stream);

    LOG(WARNING) << "dx: " << dx;
    LOG(WARNING) << "dx numel: " << dx->numel();
    LOG(WARNING) << "dx dims: " << dx->dims();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    reduce_prod, ops::ReduceProdNPUKernel<plat::NPUDeviceContext, float>,
    ops::ReduceProdNPUKernel<plat::NPUDeviceContext, plat::float16>);
REGISTER_OP_NPU_KERNEL(
    reduce_prod_grad,
    ops::ReduceProdGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::ReduceProdGradNPUKernel<plat::NPUDeviceContext, plat::float16>);
