/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename T>
inline void shift_along_dim(const T* input_data, T* output_data,
                            const DDim& input_dim, int64_t dim, int64_t shift) {
  int64_t outer_loops = 1;
  for (auto i = 0; i < dim; i++) {
    outer_loops *= input_dim[i];
  }
  auto slice_width = 1;
  for (auto i = dim + 1; i < input_dim.size(); i++) {
    slice_width *= input_dim[i];
  }
  const size_t slice_bytes = slice_width * sizeof(T);

  for (auto i = 0; i < outer_loops; i++) {
    auto input_pos = i * slice_width;
    auto output_pos = ((i + dim) % input_dim[dim]) * slice_width;
    memcpy(output_data + output_pos, input_data + input_pos, slice_bytes);
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
    std::vector<int64_t> dims = context.Attr<std::vector<int64_t>>("dims");

    PADDLE_ENFORCE_EQ(shifts.size(), dims.size(),
                      "AttrError: Attr<shifts>.size() must be equal "
                      "to Attr<dims>.size(). But received: Attr<shift>"
                      ".size = %d, Attr<dims>.size() = %d",
                      shifts.size(), dims.size());
    size_t nums = shifts.size();
    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    const DDim input_dim = input.dims();

    for (size_t i = 0; i < nums; i++) {
      shift_along_dim(input_data, output_data, input_dim, dims[i], shifts[i]);
    }
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
    std::vector<int64_t> dims = context.Attr<std::vector<int64_t>>("dims");

    PADDLE_ENFORCE_EQ(shifts.size(), dims.size(),
                      "AttrError: Attr<shifts>.size() must be equal "
                      "to Attr<dims>.size(). But received: Attr<shift>"
                      ".size = %d, Attr<dims>.size() = %d",
                      shifts.size(), dims.size());
    size_t nums = shifts.size();
    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    const DDim input_dim = input.dims();

    for (size_t i = 0; i < nums; i++) {
      shift_along_dim(input_data, output_data, input_dim, dims[i],
                      0 - shifts[i]);
    }
  }
};

}  // namespace operators
}  // namespace paddle
