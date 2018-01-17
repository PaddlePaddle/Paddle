/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SplitSelectedRowsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::SelectedRows>("X");
    auto outs = ctx.MultiOutput<framework::SelectedRows>("Out");

    auto rows_section = ctx.Attr<std::vector<int>>("rows_section");
    auto height_section = ctx.Attr<std::vector<int>>("height_section");

    int64_t n = outs.size();
    int offset = 0;

    for (int64_t i = 0; i < n; ++i) {
      framework::Vector<int64_t> out_rows;
      for (int64_t j = 0; j < rows_section[i]; ++j) {
        out_rows.push_back(x->rows()[offset + j]);
      }

      auto& out = outs[i];
      auto x_dims = x->GetCompleteDims();
      x_dims[0] = rows_section[i];
      out->mutable_value()->mutable_data<T>(x_dims, ctx.GetPlace());
      framework::Copy(x->value().Slice(offset, rows_section[i] + offset),
                      x->place(), ctx.device_context(), out->mutable_value());
      outs[i]->set_rows(out_rows);
      if (height_section.size()) {
        outs[i]->set_height(height_section[i]);
      }
      offset += rows_section[i];
    }
  }
};

}  // namespace operators
}  // namespace paddle
