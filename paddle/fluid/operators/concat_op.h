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
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    auto out_stride = framework::stride_numel(out->dims());

    size_t output_offset = 0;
    for (auto* in : ins) {
      auto in_stride = framework::stride_numel(in->dims());
      StridedNumelCopyWithAxis<T>(ctx.device_context(), axis,
                                  out->data<T>() + output_offset, out_stride,
                                  in->data<T>(), in_stride, in_stride[axis]);
      output_offset += in_stride[axis];
    }
  }
};

template <typename DeviceContext, typename T>
class ConcatGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto outs = ctx.MultiOutput<framework::Tensor>(framework::GradVarName("X"));
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    size_t input_offset = 0;
    auto in_stride = framework::stride_numel(in->dims());

    for (auto& out : outs) {
      out->mutable_data<T>(ctx.GetPlace());
      auto out_stride = framework::stride_numel(out->dims());
      StridedNumelCopyWithAxis<T>(ctx.device_context(), axis, out->data<T>(),
                                  out_stride, in->data<T>() + input_offset,
                                  in_stride, out_stride[axis]);
      input_offset += out_stride[axis];
    }
  }
};

}  // namespace operators
}  // namespace paddle
