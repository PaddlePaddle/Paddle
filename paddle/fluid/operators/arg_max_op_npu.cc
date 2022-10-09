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

#include "paddle/fluid/operators/arg_min_max_op_base.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using NPUDeviceContext = platform::NPUDeviceContext;

template <typename T>
struct VisitDataArgNPUMaxFunctor {
  const framework::ExecutionContext& ctx;

  explicit VisitDataArgNPUMaxFunctor(const framework::ExecutionContext& ctx)
      : ctx(ctx) {}
  template <typename Tout>
  void apply() const {
    auto& x = *(ctx.Input<phi::DenseTensor>("X"));
    auto& out = *(ctx.Output<phi::DenseTensor>("Out"));
    out.template mutable_data<Tout>(ctx.GetPlace());
    auto axis = ctx.Attr<int64_t>("axis");
    auto dtype = ctx.Attr<int>("dtype");
    const bool& flatten = ctx.Attr<bool>("flatten");

    Tensor transformed_x(x.type());
    transformed_x.ShareDataWith(x);
    if (flatten) {
      transformed_x.Resize(phi::make_ddim({x.numel()}));
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();
    NpuOpRunner runner;
    runner.SetType("ArgMaxV2")
        .AddInput(transformed_x)
        .AddInput(std::vector<int64_t>{axis})
        .AddOutput(out)
        .AddAttrDataType("dtype", dtype)
        .Run(stream);
  }
};

template <typename T>
class ArgMaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dtype = ctx.Attr<int>("dtype");
    if (dtype < 0) {
      framework::VisitDataTypeTiny(static_cast<framework::proto::VarType::Type>(
                                       framework::proto::VarType::INT64),
                                   VisitDataArgNPUMaxFunctor<T>(ctx));
      return;
    }
    framework::VisitDataTypeTiny(
        static_cast<framework::proto::VarType::Type>(dtype),
        VisitDataArgNPUMaxFunctor<T>(ctx));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(arg_max,
                       ops::ArgMaxNPUKernel<float>,
                       ops::ArgMaxNPUKernel<paddle::platform::float16>);
