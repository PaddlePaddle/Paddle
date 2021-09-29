/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class EqualNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("Equal", {*x, *y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class NotEqualNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("NotEqual", {*x, *y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class LessThanNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("Less", {*x, *y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class LessEqualNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<bool>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("LessEqual", {*x, *y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class GreaterThanNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    out->mutable_data<bool>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("Greater", {*x, *y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class GreaterEqualNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    out->mutable_data<bool>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("GreaterEqual", {*x, *y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    equal, ops::EqualNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::EqualNPUKernel<plat::NPUDeviceContext, float>,
    ops::EqualNPUKernel<plat::NPUDeviceContext, double>,
    ops::EqualNPUKernel<plat::NPUDeviceContext, int8_t>,
    ops::EqualNPUKernel<plat::NPUDeviceContext, uint8_t>,
    ops::EqualNPUKernel<plat::NPUDeviceContext, int16_t>,
    ops::EqualNPUKernel<plat::NPUDeviceContext, int>,
    ops::EqualNPUKernel<plat::NPUDeviceContext, int64_t>,
    ops::EqualNPUKernel<plat::NPUDeviceContext, bool>);

REGISTER_OP_NPU_KERNEL(
    not_equal, ops::NotEqualNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::NotEqualNPUKernel<plat::NPUDeviceContext, float>,
    ops::NotEqualNPUKernel<plat::NPUDeviceContext, double>,
    ops::NotEqualNPUKernel<plat::NPUDeviceContext, int8_t>,
    ops::NotEqualNPUKernel<plat::NPUDeviceContext, uint8_t>,
    ops::NotEqualNPUKernel<plat::NPUDeviceContext, int16_t>,
    ops::NotEqualNPUKernel<plat::NPUDeviceContext, int>,
    ops::NotEqualNPUKernel<plat::NPUDeviceContext, int64_t>);

REGISTER_OP_NPU_KERNEL(
    less_than, ops::LessThanNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::LessThanNPUKernel<plat::NPUDeviceContext, float>,
    ops::LessThanNPUKernel<plat::NPUDeviceContext, double>,
    ops::LessThanNPUKernel<plat::NPUDeviceContext, int8_t>,
    ops::LessThanNPUKernel<plat::NPUDeviceContext, uint8_t>,
    ops::LessThanNPUKernel<plat::NPUDeviceContext, int16_t>,
    ops::LessThanNPUKernel<plat::NPUDeviceContext, int>,
    ops::LessThanNPUKernel<plat::NPUDeviceContext, int64_t>);

REGISTER_OP_NPU_KERNEL(
    less_equal, ops::LessEqualNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::LessEqualNPUKernel<plat::NPUDeviceContext, float>,
    ops::LessEqualNPUKernel<plat::NPUDeviceContext, double>,
    ops::LessEqualNPUKernel<plat::NPUDeviceContext, int8_t>,
    ops::LessEqualNPUKernel<plat::NPUDeviceContext, uint8_t>,
    ops::LessEqualNPUKernel<plat::NPUDeviceContext, int16_t>,
    ops::LessEqualNPUKernel<plat::NPUDeviceContext, int>,
    ops::LessEqualNPUKernel<plat::NPUDeviceContext, int64_t>);

REGISTER_OP_NPU_KERNEL(
    greater_than,
    ops::GreaterThanNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::GreaterThanNPUKernel<plat::NPUDeviceContext, float>,
    ops::GreaterThanNPUKernel<plat::NPUDeviceContext, double>,
    ops::GreaterThanNPUKernel<plat::NPUDeviceContext, int8_t>,
    ops::GreaterThanNPUKernel<plat::NPUDeviceContext, uint8_t>,
    ops::GreaterThanNPUKernel<plat::NPUDeviceContext, int16_t>,
    ops::GreaterThanNPUKernel<plat::NPUDeviceContext, int>,
    ops::GreaterThanNPUKernel<plat::NPUDeviceContext, int64_t>);

REGISTER_OP_NPU_KERNEL(
    greater_equal,
    ops::GreaterEqualNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::GreaterEqualNPUKernel<plat::NPUDeviceContext, float>,
    ops::GreaterEqualNPUKernel<plat::NPUDeviceContext, double>,
    ops::GreaterEqualNPUKernel<plat::NPUDeviceContext, int8_t>,
    ops::GreaterEqualNPUKernel<plat::NPUDeviceContext, uint8_t>,
    ops::GreaterEqualNPUKernel<plat::NPUDeviceContext, int>,
    ops::GreaterEqualNPUKernel<plat::NPUDeviceContext, int64_t>);
