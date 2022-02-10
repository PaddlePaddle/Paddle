// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>
#include "concat_and_split.h"  // NOLINT
#include "paddle/extension.h"

#define CHECK_INPUT(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

int64_t ComputeAxis(int64_t axis, int64_t rank) {
  PD_CHECK(axis >= -rank && axis < rank,
           "The axis is excepted to be in range of [",
           -rank,
           ", ",
           rank,
           "].");
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

std::vector<int64_t> ComputeOutShape(
    std::vector<std::vector<int64_t>> in_shapes, int64_t axis) {
  size_t n = in_shapes.size();
  auto out_shape = in_shapes[0];
  size_t zero_dim_size = out_shape.size();
  for (size_t i = 1; i < n; ++i) {
    PD_CHECK(in_shapes[i].size() == out_shape.size(),
             "Input dimension must be same.");
    for (size_t j = 0; j < zero_dim_size; ++j) {
      if (j == axis) {
        out_shape[axis] += in_shapes[i][j];
      } else {
        PD_CHECK(in_shapes[0][j] == in_shapes[i][j],
                 "The ",
                 j,
                 "-th dimension of input must be same.");
      }
    }
  }
  return out_shape;
}

std::vector<paddle::Tensor> ConcatForwardDynamicAxis(
    const std::vector<paddle::Tensor>& inputs, const paddle::Tensor& axis_t) {
  // check inputs
  PD_CHECK(inputs.size() >= 1, "No Tensor need to be concat.");
  for (auto& t : inputs) {
    CHECK_INPUT(t);
  }
  CHECK_INPUT(axis_t);

  // compute output shape
  int64_t rank = static_cast<int64_t>(inputs[0].shape().size());
  int64_t axis = axis_t.data<int64_t>()[0];
  axis = ComputeAxis(axis, rank);
  std::vector<std::vector<int64_t>> in_shapes;
  for (auto& t : inputs) {
    in_shapes.emplace_back(t.shape());
  }
  auto out_shape = ComputeOutShape(in_shapes, axis);

  // create output
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, out_shape);

  // calc
  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
      inputs[0].type(), "ConcatCpuKernel", ([&] {
        ConcatCpuKernel<data_t>(inputs, &out, axis);
      }));

  return {out};
}

std::vector<paddle::Tensor> ConcatBackwardDynamicAxis(
    const std::vector<paddle::Tensor>& inputs,
    const paddle::Tensor& grad_out,
    const paddle::Tensor& axis_t) {
  // check input
  PD_CHECK(inputs.size() >= 1, "No Tensor need to be concat.");
  for (auto& t : inputs) {
    CHECK_INPUT(t);
  }
  CHECK_INPUT(axis_t);
  CHECK_INPUT(grad_out);

  // compate axis
  int64_t rank = static_cast<int64_t>(inputs[0].shape().size());
  int64_t axis = axis_t.data<int64_t>()[0];
  axis = ComputeAxis(axis, rank);

  // create outputs
  std::vector<paddle::Tensor> grad_inputs;
  for (auto& t : inputs) {
    auto grad = paddle::Tensor(paddle::PlaceType::kCPU, t.shape());
    grad_inputs.emplace_back(grad);
  }

  // calc
  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
      grad_out.type(), "SplitCpuKernel", ([&] {
        SplitCpuKernel<data_t>(grad_out, inputs, &grad_inputs, axis);
      }));

  return grad_inputs;
}

std::vector<std::vector<int64_t>> ConcatInferShapeDynamicAxis(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<int64_t>& axis_shape) {
  return {std::vector<int64_t>(input_shapes[0].size(), -1)};
}

std::vector<paddle::DataType> ConcatInferDtypeDynamicAxis(
    const std::vector<paddle::DataType>& input_dtypes,
    const paddle::DataType& axis_dtype) {
  return {input_dtypes[0]};
}

PD_BUILD_OP(custom_concat)
    .Inputs({paddle::Vec("X"), "Axis"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(ConcatForwardDynamicAxis))
    .SetInferShapeFn(PD_INFER_SHAPE(ConcatInferShapeDynamicAxis))
    .SetInferDtypeFn(PD_INFER_DTYPE(ConcatInferDtypeDynamicAxis));

PD_BUILD_GRAD_OP(custom_concat)
    .Inputs({paddle::Vec("X"), paddle::Grad("Out"), "Axis"})
    .Outputs({paddle::Grad(paddle::Vec("X"))})
    .SetKernelFn(PD_KERNEL(ConcatBackwardDynamicAxis));

std::vector<paddle::Tensor> ConcatForwardStaticAxis(
    const std::vector<paddle::Tensor>& inputs, const int64_t& axis) {
  // check inputs
  PD_CHECK(inputs.size() >= 1, "No Tensor need to be concat.");
  for (auto& t : inputs) {
    CHECK_INPUT(t);
  }

  // compute output shape
  int64_t rank = static_cast<int64_t>(inputs[0].shape().size());
  auto final_axis = ComputeAxis(axis, rank);
  std::vector<std::vector<int64_t>> in_shapes;
  for (auto& t : inputs) {
    in_shapes.emplace_back(t.shape());
  }
  auto out_shape = ComputeOutShape(in_shapes, final_axis);

  // create output
  auto out = paddle::Tensor(paddle::PlaceType::kCPU, out_shape);

  // calc
  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
      inputs[0].type(), "ConcatCpuKernel", ([&] {
        ConcatCpuKernel<data_t>(inputs, &out, final_axis);
      }));

  return {out};
}

std::vector<paddle::Tensor> ConcatBackwardStaticAxis(
    const std::vector<paddle::Tensor>& inputs,
    const paddle::Tensor& grad_out,
    const int64_t& axis) {
  // check input
  PD_CHECK(inputs.size() >= 1, "No Tensor need to be concat.");
  for (auto& t : inputs) {
    CHECK_INPUT(t);
  }
  CHECK_INPUT(grad_out);

  // compate axis
  int64_t rank = static_cast<int64_t>(inputs[0].shape().size());
  auto final_axis = ComputeAxis(axis, rank);

  // create outputs
  std::vector<paddle::Tensor> grad_inputs;
  for (auto& t : inputs) {
    auto grad = paddle::Tensor(paddle::PlaceType::kCPU, t.shape());
    grad_inputs.emplace_back(grad);
  }

  // calc
  PD_DISPATCH_FLOATING_AND_INTEGRAL_TYPES(
      grad_out.type(), "SplitCpuKernel", ([&] {
        SplitCpuKernel<data_t>(grad_out, inputs, &grad_inputs, final_axis);
      }));

  return grad_inputs;
}

std::vector<std::vector<int64_t>> ConcatInferShapeStaticAxis(
    const std::vector<std::vector<int64_t>>& input_shapes,
    const int64_t& axis) {
  int64_t rank = static_cast<int64_t>(input_shapes[0].size());
  auto final_axis = ComputeAxis(axis, rank);
  auto out_shape = ComputeOutShape(input_shapes, final_axis);
  return {out_shape};
}

std::vector<paddle::DataType> ConcatInferDtypeStaticAxis(
    const std::vector<paddle::DataType>& input_dtypes) {
  return {input_dtypes[0]};
}

PD_BUILD_OP(custom_concat_with_attr)
    .Inputs({paddle::Vec("X")})
    .Outputs({"Out"})
    .Attrs({"axis: int64_t"})
    .SetKernelFn(PD_KERNEL(ConcatForwardStaticAxis))
    .SetInferShapeFn(PD_INFER_SHAPE(ConcatInferShapeStaticAxis))
    .SetInferDtypeFn(PD_INFER_DTYPE(ConcatInferDtypeStaticAxis));

PD_BUILD_GRAD_OP(custom_concat_with_attr)
    .Inputs({paddle::Vec("X"), paddle::Grad("Out")})
    .Outputs({paddle::Grad(paddle::Vec("X"))})
    .Attrs({"axis: int64_t"})
    .SetKernelFn(PD_KERNEL(ConcatBackwardStaticAxis));
