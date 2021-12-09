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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CropNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");

    std::vector<int> offset_list;
    if (ctx.HasInput("Offsets")) {
      auto* offsets_tensor = ctx.Input<framework::Tensor>("Offsets");
      TensorToVector(*offsets_tensor, ctx.device_context(), &offset_list);
      if (offset_list.empty()) {
        offset_list.resize(x->dims().size(), 0);
      }
    } else {
      auto res = ctx.Attr<std::vector<int>>("offsets");
      if (res.empty()) {
        offset_list.resize(x->dims().size(), 0);
      } else {
        offset_list.insert(offset_list.end(), res.begin(), res.end());
      }
    }

    PADDLE_ENFORCE_EQ(
        static_cast<int64_t>(offset_list.size()), x->dims().size(),
        platform::errors::InvalidArgument(
            "The shape (%d) of CropOp's "
            "'offset' attribute should be equal to the shape of dims "
            "(%d) of the Input(X).",
            offset_list.size(), x->dims().size()));

    int axis_int = 0;
    framework::NPUAttributeMap attr_input = {{"offsets", offset_list},
                                             {"axis", axis_int}};
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    if (ctx.HasInput("Y")) {
      auto* shape = ctx.Input<framework::Tensor>("Y");
      PADDLE_ENFORCE_EQ(shape->dims().size(), x->dims().size(),
                        platform::errors::InvalidArgument(
                            "The shape of dims of (%d) of CropOp's "
                            "Input(shape) should be equal to the shape of dims "
                            "(%d) of the Input(X).",
                            shape->dims().size(), x->dims().size()));

      const auto& runner =
          NpuOpRunner("Crop", {*x, *shape}, {*out}, attr_input);
      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);
    } else {
      auto shape_size = ctx.Attr<std::vector<int>>("shape");
      PADDLE_ENFORCE_EQ(shape_size.size(), x->dims().size(),
                        platform::errors::InvalidArgument(
                            "The shape of dims of (%d) of CropOp's "
                            "Input(shape) should be equal to the shape of dims "
                            "(%d) of the Input(X).",
                            shape_size.size(), x->dims().size()));
      Tensor tmp_shape(x->type());
      tmp_shape.Resize(framework::make_ddim(shape_size));
      tmp_shape.mutable_data<T>(ctx.GetPlace());
      const auto& runner =
          NpuOpRunner("Crop", {*x, tmp_shape}, {*out}, attr_input);
      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    crop, ops::CropNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::CropNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::CropNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);
