
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

inline std::vector<int> get_expand_shape_npu(
    const framework::ExecutionContext& ctx) {
  std::vector<int> vec_expand_shape;
  auto list_expand_shapes_tensor =
      ctx.MultiInput<framework::Tensor>("expand_shapes_tensor");
  if (ctx.HasInput("Shape")) {
    auto* shape_tensor = ctx.Input<framework::LoDTensor>("Shape");
    std::vector<int> out_data;
    TensorToVector(*shape_tensor, ctx.device_context(), &out_data);
    for (int i = 0; i < static_cast<int>(out_data.size()); ++i) {
      vec_expand_shape.push_back(out_data[i]);
    }
    return vec_expand_shape;
  } else if (list_expand_shapes_tensor.size() > 0) {
    // get tensor from
    for (size_t i = 0; i < list_expand_shapes_tensor.size(); ++i) {
      auto tensor = list_expand_shapes_tensor[i];
      std::vector<int> out_data;
      TensorToVector(*tensor, ctx.device_context(), &out_data);
      vec_expand_shape.push_back(out_data[0]);
    }
    return vec_expand_shape;
  } else {
    return ctx.Attr<std::vector<int>>("shape");
  }
}

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class ExpandV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Out = ctx.Output<framework::Tensor>("Out");

    std::vector<int> expand_shape = get_expand_shape_npu(ctx);

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

template <typename DeviceContext, typename T>
class ExpandV2NPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // case 1: reduce dout dims to dx dims
    // For example: [2, 120] --> [120]
    auto reduce_ndim = dout->dims().size() - dx->dims().size();
    std::vector<int> axes;
    for (auto i = 0; i < reduce_ndim; ++i) {
      axes.push_back(i);
    }
    Tensor* tmp_dout = const_cast<Tensor*>(dout);
    Tensor reduced_dout(dx->type());
    if (axes.size() != 0) {
      std::vector<int64_t> reduced_dout_dims;
      for (auto i = reduce_ndim; i < dout->dims().size(); ++i) {
        reduced_dout_dims.push_back(dout->dims()[i]);
      }
      reduced_dout.Resize(framework::make_ddim(reduced_dout_dims));
      reduced_dout.mutable_data<T>(ctx.GetPlace());
      const auto& runner = NpuOpRunner("ReduceSumD", {*dout}, {reduced_dout},
                                       {{"axes", axes}, {"keep_dims", false}});
      runner.Run(stream);
      tmp_dout = &reduced_dout;
    }

    // case 2: reduce axis of dout in which dim is 1
    // For example: [12, 140] --> [1, 140]

    // case 3: copy dout to dx when shape is totally same, and dim in dx != 1
    // For example: [2, 10, 5] --> [2, 10, 5]
    axes.clear();
    for (auto i = 0; i < dx->dims().size(); ++i) {
      if (dx->dims()[i] == 1) {
        axes.push_back(i);
      }
    }
    if (axes.size() != 0) {
      const auto& runner = NpuOpRunner("ReduceSumD", {*tmp_dout}, {*dx},
                                       {{"axes", axes}, {"keep_dims", true}});
      runner.Run(stream);
    } else {
      framework::TensorCopySync(*tmp_dout, ctx.GetPlace(), dx);
    }
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

REGISTER_OP_NPU_KERNEL(
    expand_v2_grad,
    ops::ExpandV2NPUGradKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ExpandV2NPUGradKernel<paddle::platform::NPUDeviceContext, int>);
