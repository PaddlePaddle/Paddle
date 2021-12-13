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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static inline std::vector<int> GetPaddings(
    const framework::ExecutionContext& context) {
  std::vector<int> paddings(6);
  auto* paddings_t = context.Input<Tensor>("Paddings");
  if (paddings_t) {
    TensorToVector(*paddings_t, context.device_context(), &paddings);
  } else {
    auto pads = context.Attr<std::vector<int>>("paddings");
    std::copy(pads.begin(), pads.end(), paddings.data());
  }
  return paddings;
}

template <typename T>
class Pad3dNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto in_dims = x->dims();

    std::vector<int> pads = GetPaddings(context);
    auto mode = context.Attr<std::string>("mode");
    float value = context.Attr<float>("value");
    auto data_format = context.Attr<std::string>("data_format");

    auto* out = context.Output<Tensor>("Out");

    PADDLE_ENFORCE_LT(abs(value), 1e-5,
                      platform::errors::Unimplemented(
                          "Ascend npu only support constant_values=0 right now,"
                          "but received constant_value is %f .",
                          value));

    PADDLE_ENFORCE_EQ(mode, "constant",
                      platform::errors::Unimplemented(
                          "Ascend npu only support mode=constant right now,"
                          "but received mode is %s .",
                          mode));

    std::vector<int> paddings(
        {0, 0, 0, 0, pads[4], pads[5], pads[2], pads[3], pads[0], pads[1]});
    if (data_format == "NCDHW") {
      out->Resize({in_dims[0], in_dims[1], in_dims[2] + pads[4] + pads[5],
                   in_dims[3] + pads[2] + pads[3],
                   in_dims[4] + pads[0] + pads[1]});
    } else {
      out->Resize({in_dims[0], in_dims[1] + pads[4] + pads[5],
                   in_dims[2] + pads[2] + pads[3],
                   in_dims[3] + pads[0] + pads[1], in_dims[4]});
      paddings = {0,       0,       pads[4], pads[5], pads[2],
                  pads[3], pads[0], pads[1], 0,       0};
    }
    out->mutable_data<T>(context.GetPlace());

    NpuOpRunner runner;
    runner.SetType("PadV3")
        .AddInput(*x)
        .AddInput(std::move(paddings))
        .AddInput(
            std::vector<int>({0}))  // npu only support constant_value=0 now
        .AddOutput(*out)
        .AddAttr("mode", mode);

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename T>
class Pad3dGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    std::vector<int> pads = GetPaddings(context);
    auto mode = context.Attr<std::string>("mode");
    auto data_format = context.Attr<std::string>("data_format");

    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_in = context.Output<Tensor>(framework::GradVarName("X"));
    auto d_in_dims = d_in->dims();
    d_in->mutable_data<T>(context.GetPlace());

    const int pad_left = pads[0];
    const int pad_top = pads[2];
    const int pad_front = pads[4];

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<int64_t> size(
        {d_in_dims[0], d_in_dims[1], d_in_dims[2], d_in_dims[3], d_in_dims[4]});
    if (mode == "constant") {  // this method can be only used for constant mode
      std::vector<int> offsets({0, 0, pad_front, pad_top, pad_left});
      if (data_format == "NDHWC") {
        offsets = {0, pad_front, pad_top, pad_left, 0};
      }
      const auto& runner = NpuOpRunner("SliceD", {*d_out}, {*d_in},
                                       {{"offsets", offsets}, {"size", size}});
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(pad3d, ops::Pad3dNPUKernel<plat::float16>,
                       ops::Pad3dNPUKernel<float>, ops::Pad3dNPUKernel<int>);

REGISTER_OP_NPU_KERNEL(pad3d_grad, ops::Pad3dNPUKernel<plat::float16>,
                       ops::Pad3dGradNPUKernel<float>);
