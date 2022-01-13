/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class PadNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    auto paddings = context.Attr<std::vector<int>>("paddings");
    float pad_value = context.Attr<float>("pad_value");

    PADDLE_ENFORCE_LT(abs(pad_value), 1e-5,
                      platform::errors::Unimplemented(
                          "Ascend npu only support pad_value=0 right now,"
                          "but received pad_value is %f .",
                          pad_value));

    out->mutable_data<T>(context.GetPlace());

    NpuOpRunner runner;
    runner.SetType("Pad")
        .AddInput(*x)
        .AddInput(std::move(paddings))
        .AddOutput(*out);

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename T>
class PadGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto paddings = context.Attr<std::vector<int>>("paddings");

    d_x->mutable_data<T>(context.GetPlace());

    auto d_x_dims = d_x->dims();
    auto size = paddle::framework::vectorize(d_x_dims);
    std::vector<int> offsets(0);
    int i = 0;
    for (auto iter = paddings.begin(); iter < paddings.end(); ++iter, ++i) {
      if (i % 2 == 0) {
        offsets.push_back(*iter);
      }
    }

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("SliceD", {*d_out}, {*d_x},
                                     {{"offsets", offsets}, {"size", size}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(pad, ops::PadNPUKernel<plat::float16>,
                       ops::PadNPUKernel<float>, ops::PadNPUKernel<int>);

REGISTER_OP_NPU_KERNEL(pad_grad, ops::PadNPUKernel<plat::float16>,
                       ops::PadGradNPUKernel<float>);
