
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

#include "paddle/fluid/operators/expand_v2_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class ExpandV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Out = ctx.Output<framework::Tensor>("Out");

    std::vector<int> expand_shape;
    auto list_expand_shapes_tensor =
        ctx.MultiInput<framework::Tensor>("expand_shapes_tensor");
    if (ctx.HasInput("Shape")) {
      auto* shape_tensor = ctx.Input<framework::LoDTensor>("Shape");
      std::vector<int> out_data;
      TensorToVector(*shape_tensor, ctx.device_context(), &out_data);
      for (int i = 0; i < static_cast<int>(out_data.size()); ++i) {
        expand_shape.push_back(out_data[i]);
      }
    } else if (list_expand_shapes_tensor.size() > 0) {
      // get tensor from
      for (size_t i = 0; i < list_expand_shapes_tensor.size(); ++i) {
        auto tensor = list_expand_shapes_tensor[i];
        std::vector<int> out_data;
        TensorToVector(*tensor, ctx.device_context(), &out_data);
        expand_shape.push_back(out_data[0]);
      }
    } else {
      expand_shape = ctx.Attr<std::vector<int>>("shape");
    }

    framework::NPUAttributeMap attr_input = {{"shape", expand_shape}};

    auto rank = X->dims().size();
    for (size_t i = 0; i < expand_shape.size(); ++i) {
      PADDLE_ENFORCE_GT(
          expand_shape[i], 0,
          platform::errors::InvalidArgument(
              "The %uth element of 'shape' for expand_v2_npu op must be "
              "greater than 0, but the value given is %d.",
              i, expand_shape[i]));
    }
    PADDLE_ENFORCE_GE(
        rank, 1,
        platform::errors::InvalidArgument(
            "The rank of the input 'X' for expand_v2_npu op must be positive, "
            "but the value received is %d.",
            rank));
    PADDLE_ENFORCE_LE(
        rank, MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The rank of the input 'X' for expand_v2_npu op must be less than "
            "or equal to %d, but the value received is %d.",
            MAX_RANK_SUPPORTED, rank));
    auto shape_size = expand_shape.size();
    PADDLE_ENFORCE_GE(
        shape_size, rank,
        platform::errors::InvalidArgument(
            "The number (%d) of elements of 'shape' for expand_v2_npu op must "
            "be "
            "greater than or equal to the rank (%d) of the input 'X'.",
            shape_size, rank));
    PADDLE_ENFORCE_LE(shape_size, MAX_RANK_SUPPORTED,
                      platform::errors::InvalidArgument(
                          "The number (%d) of elements of 'shape' for "
                          "expand_v2_npu op must be "
                          "less than or equal to %d.",
                          shape_size, MAX_RANK_SUPPORTED));

    framework::DDim out_dims = framework::make_ddim(expand_shape);
    Out->Resize(out_dims);
    Out->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("ExpandD", {*X}, {*Out}, attr_input);
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
    expand_v2,
    ops::ExpandV2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ExpandV2NPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>,
    ops::ExpandV2NPUKernel<paddle::platform::NPUDeviceContext, int>);
