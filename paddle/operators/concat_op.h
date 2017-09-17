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
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class ConcatKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    size_t n = ins.size();
    size_t output_axis_dim = 0;
    size_t before = 1, after = 1;
    for (size_t i = 0; i < n; i++) {
      output_axis_dim += ins[i]->dims()[axis];
    }
    auto& input_zero = ins[0];
    for (int64_t i = 0; i < input_zero->dims().size(); i++) {
      if (i == axis) {
        continue;
      }
      if (i < axis) {
        before *= input_zero->dims()[i];
      } else {
        after *= input_zero->dims()[i];
      }
    }
    size_t output_offset = 0;
    for (size_t i = 0; i < n; i++) {
      auto& in = ins[i];
      auto axis_dim = in->dims()[axis];
      math::copy_matrix<Place, T>(in, axis_dim, out, output_axis_dim,
                                  axis_dim * after * sizeof(T), after, before,
                                  output_offset);
      output_offset += axis_dim * after;
    }
  }
};

template <typename Place, typename T>
class ConcatGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto outs = ctx.MultiOutput<framework::Tensor>(framework::GradVarName("X"));
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    size_t before = 1, after = 1;
    size_t n = outs.size();
    size_t input_axis_dim = 0;
    for (size_t i = 0; i < n; i++) {
      input_axis_dim += outs[i]->dims()[axis];
    }
    for (int64_t i = 0; i < in->dims().size(); ++i) {
      if (i == axis) {
        continue;
      }
      if (i < axis) {
        before *= in->dims()[i];
      } else {
        after *= in->dims()[i];
      }
    }
    size_t input_offset = 0;
    for (size_t i = 0; i < n; i++) {
      auto& out = outs[i];
      size_t axis_dim = out->dims()[axis];
      printf("i->%ld before->%ld after->%ld\n", i, before, after);
      math::copy_matrix<Place, T>(in, axis_dim, out, input_axis_dim,
                                  axis_dim * after * sizeof(T), after, before,
                                  input_offset);
      /**
      for (size_t j = 0; j < before; j++) {
        size_t len = axis_dim * after * sizeof(T);
        T* dest =
            out->mutable_data<T>(platform::CPUPlace()) + axis_dim * after * j;
        const T* src =
            in->data<T>() + input_offset + input_axis_dim * after * j;
        memcpy(dest, src, len);
      }**/
      input_offset += axis_dim * after;
    }
  }
};

}  // namespace operators
}  // namespace paddle
