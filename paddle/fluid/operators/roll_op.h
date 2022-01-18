// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <memory>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename T>
inline void shift_along_dim(T* data, const DDim& input_dim, int64_t dim,
                            int64_t shift) {
  if (dim < 0) {
    dim += input_dim.size();
  }
  if (input_dim[dim] == 0) {
    return;
  }
  shift = shift % input_dim[dim];
  if (shift < 0) {
    shift += input_dim[dim];
  }

  auto outer_loops = 1;
  for (auto i = 0; i < dim; i++) {
    outer_loops *= input_dim[i];
  }
  auto slice_width = 1;
  for (auto i = dim + 1; i < input_dim.size(); i++) {
    slice_width *= input_dim[i];
  }

  VLOG(3) << "shift_along_dim_debug: input_dim: " << input_dim
          << "; dim: " << dim << "; shift: " << shift
          << "; outer_loops: " << outer_loops
          << "; slice_width: " << slice_width;
  if (shift == 0) {
    return;
  }

  std::vector<T> head;
  auto head_size = slice_width * (input_dim[dim] - shift);
  head.resize(head_size);

  for (auto i = 0; i < outer_loops; i++) {
    for (auto j = 0; j < head_size; j++) {
      head[j] = data[i * input_dim[dim] * slice_width + j];
    }
    for (auto j = input_dim[dim] - shift; j < input_dim[dim]; j++) {
      auto dst_pos = j - input_dim[dim] + shift;
      for (auto k = 0; k < slice_width; k++) {
        data[(i * input_dim[dim] + dst_pos) * slice_width + k] =
            data[(i * input_dim[dim] + j) * slice_width + k];
      }
    }
    for (auto j = 0; j < head_size; j++) {
      data[(i * input_dim[dim] + shift) * slice_width + j] = head[j];
    }
  }
}

template <typename DeviceContext, typename T>
class RollKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_var = context.InputVar("X");
    auto* output_var = context.OutputVar("Out");
    auto& input = input_var->Get<LoDTensor>();
    auto* output = output_var->GetMutable<LoDTensor>();
    std::vector<int64_t> shifts = context.Attr<std::vector<int64_t>>("shifts");
    if (context.HasInput("ShiftsTensor")) {
      const auto* shifts_tensor =
          context.Input<framework::Tensor>("ShiftsTensor");
      PADDLE_ENFORCE_EQ(
          shifts_tensor->dims().size(), 1,
          platform::errors::InvalidArgument(
              "The rank of ShiftsTensor is expected to be 1, got %s",
              shifts_tensor->dims().size()));
      shifts = GetDataFromTensor<int64_t>(shifts_tensor);
    }
    std::vector<int64_t> dims = context.Attr<std::vector<int64_t>>("axis");

    std::vector<T> out_vec;
    paddle::framework::TensorToVector(input, context.device_context(),
                                      &out_vec);

    size_t nums = shifts.size();
    DDim input_dim = input.dims();

    // axis = none, reshape to 1-D tensor
    if (dims.size() == 0) {
      dims.push_back(0l);
      input_dim = framework::Dim<1>(out_vec.size());
    }

    for (size_t i = 0; i < nums; i++) {
      PADDLE_ENFORCE_EQ(
          dims[i] < input_dim.size() && dims[i] >= (0 - input_dim.size()), true,
          platform::errors::OutOfRange(
              "Attr(axis[%d]) is out of range, It's expected "
              "to be in range of [-%d, %d]. But received Attr(axis[%d]) = %d.",
              i, input_dim.size(), input_dim.size() - 1, i, dims[i]));
      shift_along_dim(out_vec.data(), input_dim, dims[i], shifts[i]);
    }
    output->mutable_data<T>(context.GetPlace());
    framework::TensorFromVector(out_vec, context.device_context(), output);
    output->Resize(input.dims());
  }
};

template <typename DeviceContext, typename T>
class RollGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input_var = context.InputVar(framework::GradVarName("Out"));
    auto* output_var = context.OutputVar(framework::GradVarName("X"));
    auto& input = input_var->Get<LoDTensor>();
    auto* output = output_var->GetMutable<LoDTensor>();
    std::vector<int64_t> shifts = context.Attr<std::vector<int64_t>>("shifts");
    if (context.HasInput("ShiftsTensor")) {
      const auto* shifts_tensor =
          context.Input<framework::Tensor>("ShiftsTensor");
      shifts = GetDataFromTensor<int64_t>(shifts_tensor);
    }
    std::vector<int64_t> dims = context.Attr<std::vector<int64_t>>("axis");

    std::vector<T> out_vec;
    paddle::framework::TensorToVector(input, context.device_context(),
                                      &out_vec);

    size_t nums = shifts.size();
    DDim input_dim = input.dims();

    // axis = none, reshape to 1-D tensor
    if (dims.size() == 0) {
      dims.push_back(0l);
      input_dim = framework::Dim<1>(out_vec.size());
    }

    for (size_t i = 0; i < nums; i++) {
      shift_along_dim(out_vec.data(), input_dim, dims[i], 0 - shifts[i]);
    }
    output->mutable_data<T>(context.GetPlace());
    framework::TensorFromVector(out_vec, context.device_context(), output);
    output->Resize(input.dims());
  }
};

}  // namespace operators
}  // namespace paddle
