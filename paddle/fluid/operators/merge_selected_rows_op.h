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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

static std::vector<int64_t> ToAbsoluteSection(
    const std::vector<int64_t>& height_sections) {
  std::vector<int64_t> abs_sections;
  abs_sections.resize(height_sections.size());
  abs_sections[0] = 0;
  for (size_t i = 1; i < height_sections.size(); ++i) {
    abs_sections[i] = height_sections[i - 1] + abs_sections[i - 1];
  }
  return abs_sections;
}

template <typename DeviceContext, typename T>
class MergeSelectedRowsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto xs = ctx.MultiInput<framework::SelectedRows>("X");
    auto* out = ctx.Output<framework::SelectedRows>("Out");
    out->mutable_rows()->clear();
    auto height_sections = ctx.Attr<std::vector<int64_t>>("height_sections");

    auto abs_sections = ToAbsoluteSection(height_sections);

    out->set_height(abs_sections.back());

    size_t in_size = xs.size();
    PADDLE_ENFORCE(in_size, abs_sections.size(),
                   "input num should be the same with abs_sections");

    std::vector<framework::SelectedRows> abs_row_inputs;
    abs_row_inputs.resize(in_size);
    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* in = xs[i];
      if (!in->rows().empty()) {
        auto& abs_in = abs_row_inputs[i];
        abs_in.mutable_value()->ShareDataWith(in->value());

        std::vector<int64_t> abs_row;
        for (auto id : in->rows()) {
          abs_row.push_back(id + abs_sections[i]);
        }
        abs_in.set_rows(abs_row);
      }
    }

    std::vector<const paddle::framework::SelectedRows*> inputs;
    for (auto& in : abs_row_inputs) {
      if (!in.rows().empty()) {
        inputs.push_back(&in);
      }
    }
    math::scatter::MergeAdd<DeviceContext, T> merge_add;
    merge_add(ctx.template device_context<DeviceContext>(), inputs, out);
  }
};

}  // namespace operators
}  // namespace paddle
