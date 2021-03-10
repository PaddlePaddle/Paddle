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
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

template <typename T>
class ShapeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_var = ctx.InputVar("Input");
    auto axes = ctx.Attr<std::vector<int>>("axes");
    framework::DDim in_dims, out_dims;
    if (in_var->IsType<SelectedRows>()) {
      in_dims = in_var->Get<SelectedRows>().value().dims();
    } else {
      in_dims = in_var->Get<LoDTensor>().dims();
    }

    if (axes.empty()) {
      out_dims = in_dims;
    } else {
      std::vector<int> out_vec_dims(axes.size());
      for (size_t i = 0; i < axes.size(); ++i) {
        int axis = axes[i];
        if (axis < 0) {
          axis += in_dims.size();
        }
        PADDLE_ENFORCE_LT(static_cast<int>(axis), in_dims.size(),
                          platform::errors::InvalidArgument(
                              "The index of dimension in axes must be less "
                              "than the size of input shape."));
        out_vec_dims[i] = axis;
      }
      out_dims = framework::make_ddim(out_vec_dims);
    }

    auto* out_t = ctx.Output<Tensor>("Out");
    out_t->Resize({out_dims.size()});
    auto out_data = out_t->mutable_data<int32_t>(platform::CPUPlace());
    for (int i = 0; i < out_dims.size(); ++i) {
      if (axes.empty()) {
        out_data[i] = in_dims[i];
      } else {
        if (axes[i] < 0) {
          out_data[i] = in_dims[axes[i] + in_dims.size()];
        } else {
          out_data[i] = in_dims[axes[i]];
        }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
