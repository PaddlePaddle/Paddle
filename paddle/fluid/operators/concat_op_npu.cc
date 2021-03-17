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

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {
static inline framework::DDim ComputeAndCheckShape(
    const bool is_runtime, const std::vector<framework::DDim>& inputs_dims,
    const size_t axis) {
  const size_t n = inputs_dims.size();
  auto out_dims = inputs_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    PADDLE_ENFORCE_EQ(inputs_dims[i].size(), out_dims.size(),
                      platform::errors::InvalidArgument(
                          "The shape of input[0] and input[%d] "
                          "is expected to be equal."
                          "But received input[0]'s shape = "
                          "[%s], input[%d]'s shape = [%s].",
                          i, inputs_dims[0], i, inputs_dims[i]));
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        if (is_runtime) {
          out_dims[axis] += inputs_dims[i][j];
        } else {
          if (inputs_dims[i][j] == -1 || out_dims[j] == -1) {
            out_dims[axis] = -1;
          } else {
            out_dims[axis] += inputs_dims[i][j];
          }
        }
      } else {
        bool check_shape =
            is_runtime || (inputs_dims[0][j] > 0 && inputs_dims[i][j] > 0);
        if (check_shape) {
          // check all shape in run time
          PADDLE_ENFORCE_EQ(inputs_dims[0][j], inputs_dims[i][j],
                            platform::errors::InvalidArgument(
                                "The %d-th dimension of input[0] and input[%d] "
                                "is expected to be equal."
                                "But received input[0]'s shape = "
                                "[%s], input[%d]'s shape = [%s].",
                                j, i, inputs_dims[0], i, inputs_dims[i]));
        }
        if (!is_runtime && out_dims[j] == -1 && inputs_dims[i][j] > 0) {
          out_dims[j] = inputs_dims[i][j];
        }
      }
    }
  }
  return out_dims;
}

static inline int64_t ComputeAxis(int64_t axis, int64_t rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank, true,
      platform::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d", -rank,
          rank, axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

template <typename DeviceContext, typename T>
class ConcatNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    framework::LoDTensor* out = ctx.Output<framework::LoDTensor>("Out");
    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));
    auto axis = ctx.Attr<int>("axis");

    if (ctx.HasInput("AxisTensor")) {
      PADDLE_THROW(platform::errors::NotFound(
          "The AxisTensor is not supported on NPU now."));
    }
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));

    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    std::vector<framework::Tensor> inputs;
    for (size_t j = 0; j < ins.size(); ++j) {
      if (ins[j] && ins[j]->numel() > 0) {
        inputs.push_back(*ins[j]);
      } else {
        continue;
      }
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto runner =
        NpuOpRunner("ConcatV2D", inputs, {*out}, {{"concat_dim", axis}});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ConcatGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    auto out_var_names = ctx.OutputNames(framework::GradVarName("X"));
    auto outs =
        ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));

    {
      auto dx = outs;
      auto x = ins;
      for (size_t i = 0; i < dx.size(); ++i) {
        if (dx[i] != nullptr) {
          dx[i]->set_lod(x[i]->lod());
        }
      }
    }
    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));

    auto axis = ctx.Attr<int>("axis");

    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));
    // get output tensor that the name is not kEmptyVarName
    std::vector<framework::Tensor> outputs;
    for (size_t j = 0; j < outs.size(); ++j) {
      if (out_var_names[j] != framework::kEmptyVarName &&
          outs[j]->numel() != 0UL) {
        outs[j]->mutable_data<T>(ctx.GetPlace());
        outputs.push_back((*outs[j]);
      }
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto runner =
        NpuOpRunner("SplitD", {*out_grad}, {outputs},
                    {{"split_dim", axis}, {"num_split", outputs.size()}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    ops::ConcatNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ConcatNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::ConcatNPUKernel<paddle::platform::NPUDeviceContext,
                         paddle::platform::float16>,
    ops::ConcatNPUKernel<paddle::platform::NPUDeviceContext, int>);
REGISTER_OP_NPU_KERNEL(
    concat_grad,
    ops::ConcatGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ConcatGradNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::ConcatGradNPUKernel<paddle::platform::NPUDeviceContext,
                             paddle::platform::float16>,
    ops::ConcatGradNPUKernel<paddle::platform::NPUDeviceContext, int>);
