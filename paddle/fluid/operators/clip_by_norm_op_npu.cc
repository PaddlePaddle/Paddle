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

#include "paddle/fluid/operators/clip_by_norm_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class NPUClipByNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max_norm = context.Attr<float>("max_norm");
    auto in_var = context.InputVar("X");

    if (!(in_var->IsType<framework::LoDTensor>())) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid input variable type, only support LodTensor"
          "type, but got type is %s.",
          framework::ToTypeName(in_var->Type())));
    }

    auto place = context.GetPlace();
    auto& dev_ctx =
        context.template device_context<paddle::platform::NPUDeviceContext>();
    auto stream = dev_ctx.stream();

    auto* input = context.Input<phi::DenseTensor>("X");
    auto* output = context.Output<phi::DenseTensor>("Out");
    output->mutable_data<T>(place);

    PADDLE_ENFORCE_NOT_NULL(input,
                            platform::errors::InvalidArgument(
                                "Input(X) of ClipByNormOp should not be null. "
                                "Please check if it is created correctly."));

    Tensor square_sum(input->type());
    square_sum.mutable_data<T>(framework::DDim({1}), place);
    const auto& x_dims = input->dims();
    std::vector<int> axis;
    for (int i = 0; i < x_dims.size(); ++i) {
      axis.push_back(i);
    }
    const auto& square_sum_runner =
        NpuOpRunner("SquareSumV1",
                    {*input},
                    {square_sum},
                    {{"axis", axis}, {"keep_dims", false}});
    square_sum_runner.Run(stream);

    Tensor x_norm(input->type());
    x_norm.mutable_data<T>(framework::DDim({1}), place);
    const auto& x_norm_runner = NpuOpRunner("Sqrt", {square_sum}, {x_norm}, {});
    x_norm_runner.Run(stream);

    Tensor x_norm_t;
    framework::TensorCopySync(x_norm, platform::CPUPlace(), &x_norm_t);
    auto x_norm_v = static_cast<float>(*x_norm_t.data<T>());
    if (x_norm_v <= max_norm) {
      framework::TensorCopy(*input, place, dev_ctx, output);
    } else {
      auto epsilon = x_norm_v <= static_cast<float>(1e-30)
                         ? static_cast<float>(1e-6)
                         : static_cast<float>(0);
      float scaling = max_norm / (x_norm_v + epsilon);
      const auto& muls_runner =
          NpuOpRunner("Muls", {*input}, {*output}, {{"value", scaling}});
      muls_runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    clip_by_norm,
    ops::NPUClipByNormKernel<paddle::platform::NPUDeviceContext, float>,
    ops::NPUClipByNormKernel<paddle::platform::NPUDeviceContext,
                             plat::float16>);
