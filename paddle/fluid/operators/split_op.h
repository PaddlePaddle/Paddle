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

#include <chrono>  // NOLINT
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {
static inline std::vector<framework::DDim> UpdateOutsDims(
    const bool is_runtime, const framework::DDim in_dims, const size_t num,
    const std::vector<int>& sections, const size_t axis,
    const int outs_number) {
  std::vector<framework::DDim> outs_dims(outs_number, in_dims);
  int64_t input_axis_dim = in_dims[axis];
  if (num > 0) {
    if (is_runtime || input_axis_dim > 0) {
      PADDLE_ENFORCE_EQ(input_axis_dim % num, 0,
                        "tensor split does not result"
                        " in an equal division");
      size_t out_axis_dim = input_axis_dim / num;

      for (auto& out_dim : outs_dims) {
        out_dim[axis] = out_axis_dim;
      }
    } else {
      for (auto& out_dim : outs_dims) {
        out_dim[axis] = -1;
      }
    }
  } else if (sections.size() > 0) {
    bool all_positive = std::all_of(sections.cbegin(), sections.cend(),
                                    [](int i) { return i > 0; });
    if (is_runtime || (input_axis_dim > 0 && all_positive)) {
      int sum_of_section = 0;
      for (int section : sections) {
        sum_of_section += section;
      }
      PADDLE_ENFORCE_EQ(
          sum_of_section, input_axis_dim,
          "Sum of Attr(num_or_sections) must be equal to the input's size "
          "along the split dimension. But received Attr(num_or_sections)"
          " = [%s], input(X)'s shape = [%s], Attr(dim) = %d, ",
          framework::make_ddim(sections), in_dims, axis);
    }
    for (size_t i = 0; i < outs_number; ++i) {
      outs_dims[i][axis] = sections[i];
    }
  }

  return outs_dims;
}
template <typename DeviceContext, typename T>
class SplitOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    int num = ctx.Attr<int>("num");
    std::vector<int> sections = ctx.Attr<std::vector<int>>("sections");
    int axis = ctx.Attr<int>("axis");

    auto in_dims = in->dims();
    auto outs_number = outs.size();

    if (ctx.HasInput("AxisTensor")) {
      std::vector<framework::DDim> outs_dims(outs_number, in_dims);
      auto* axis_tensor = ctx.Input<framework::Tensor>("AxisTensor");
      axis = GetDataFromTensor<int>(axis_tensor)[0];

      outs_dims =
          UpdateOutsDims(true, in_dims, num, sections, axis, outs_number);
      for (size_t j = 0; j < outs.size(); ++j) {
        outs[j]->Resize(outs_dims[j]);
      }
    }
    auto place = ctx.GetPlace();

    std::vector<const framework::Tensor*> shape_refer;
    for (size_t j = 0; j < outs.size(); ++j) {
      outs[j]->mutable_data<T>(ctx.GetPlace());
      shape_refer.emplace_back(outs[j]);
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    // Sometimes direct copies will be faster, this maybe need deeply analysis.
    if (axis == 0 && outs.size() < 10) {
      StridedMemcpyWithAxis0<T>(dev_ctx, *in, shape_refer, &outs);
    } else {
      math::SplitFunctor<DeviceContext, T> functor;
      functor(dev_ctx, *in, shape_refer, axis, &outs);
    }
  }
};

class SplitGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto op = new framework::OpDesc();
    op->SetType("concat");
    op->SetInput("X", OutputGrad("Out"));
    op->SetInput("AxisTensor", Input("AxisTensor"));
    op->SetOutput("Out", InputGrad("X"));
    op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(op);
  }
};

}  // namespace operators
}  // namespace paddle
