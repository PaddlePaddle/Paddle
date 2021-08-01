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

#include "paddle/fluid/operators/crop_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include<iostream>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CropNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");

    // auto* Offset = ctx.Input<framework::Tensor>("Offsets");
    std::vector<int> Offset_list;
    if (ctx.HasInput("Offsets")) {
      auto* offsets_tensor = ctx.Input<framework::Tensor>("Offsets");
      TensorToVector(*offsets_tensor, ctx.device_context(), &Offset_list);
      if(Offset_list.empty()){
        Offset_list.resize(X->dims().size(),0);
      }
    }else{
      auto res = ctx.Attr<std::vector<int>>("offsets");
      if(res.empty()){
        Offset_list.resize(X->dims().size(),0);
      }else{
        Offset_list = res;
      }
    }

    PADDLE_ENFORCE_EQ(
          int64_t(Offset_list.size()), X->dims().size(),
          platform::errors::InvalidArgument(
              "The shape (%d) of CropOp's "
              "'offset' attribute should be equal to the shape of dims "
              "(%d) of the Input(X).",
              Offset_list.size(), X->dims().size()));
    
    int axis_int = 0;
    framework::NPUAttributeMap attr_input = {{"offsets", Offset_list},{"axis", axis_int}};
    auto* Out = ctx.Output<framework::Tensor>("Out");
    Out->mutable_data<T>(ctx.GetPlace());

    auto* Shape = ctx.Input<framework::Tensor>("Y");
    PADDLE_ENFORCE_EQ(
          Shape->dims().size(), X->dims().size(),
          platform::errors::InvalidArgument(
              "The shape of dims of (%d) of CropOp's "
              "Input(shape) should be equal to the shape of dims "
              "(%d) of the Input(X).",
              Shape->dims().size(), X->dims().size()));

    const auto& runner = NpuOpRunner("Crop", {*X, *Shape}, {*Out}, attr_input);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
    
    
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    crop,
    ops::CropNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::CropNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::CropNPUKernel<paddle::platform::NPUDeviceContext,  paddle::platform::float16>);
