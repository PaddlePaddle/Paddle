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

#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

static inline int64_t ComputeAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

template <typename DeviceContext, typename T>
class ConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    framework::Tensor* out = ctx.Output<framework::Tensor>("Out");
    PADDLE_ENFORCE(ins[0], "The input should not be null.");
    auto axis = ComputeAxis(static_cast<int64_t>(ctx.Attr<int>("axis")),
                            static_cast<int64_t>(ins[0]->dims().size()));
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    // Sometimes direct copies will be faster, this maybe need deeply analysis.
    if (axis == 0 && ins.size() < 10) {
      size_t output_offset = 0;
      for (auto* in : ins) {
        if (!in || in->numel() == 0UL) {
          continue;
        }
        auto in_stride = framework::stride_numel(in->dims());
        auto out_stride = framework::stride_numel(out->dims());
        StridedNumelCopyWithAxis<T>(ctx.device_context(), axis,
                                    out->data<T>() + output_offset, out_stride,
                                    in->data<T>(), in_stride, in_stride[axis]);
        output_offset += in_stride[axis];
      }
    } else {
      std::vector<framework::Tensor> inputs;
      for (size_t j = 0; j < ins.size(); ++j) {
        if (ins[j] && ins[j]->numel() > 0) {
          inputs.push_back(*ins[j]);
        } else {
          continue;
        }
      }
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      paddle::operators::math::ConcatFunctor<DeviceContext, T> concat_functor;
      concat_functor(dev_ctx, inputs, static_cast<int>(axis), out);
    }
  }
};

template <typename DeviceContext, typename T>
class ConcatGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    auto out_var_names = ctx.Outputs(framework::GradVarName("X"));
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
    PADDLE_ENFORCE(ins[0], "The input should not be null.");
    auto axis = ComputeAxis(static_cast<int64_t>(ctx.Attr<int>("axis")),
                            static_cast<int64_t>(ins[0]->dims().size()));

    // get output tensor that the name is not kEmptyVarName
    std::vector<framework::Tensor*> outputs;
    for (size_t j = 0; j < outs.size(); ++j) {
      if (out_var_names[j] != framework::kEmptyVarName &&
          outs[j]->numel() != 0UL) {
        outs[j]->mutable_data<T>(ctx.GetPlace());
        outputs.push_back(outs[j]);
      } else {
        outputs.push_back(nullptr);
      }
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // Sometimes direct copies will be faster, this maybe need deeply analysis.
    if (axis == 0 && outs.size() < 10) {
      std::vector<const framework::Tensor*> ref_shape;
      ref_shape.insert(ref_shape.begin(), ins.begin(), ins.end());
      StridedMemcpyWithAxis0<T>(dev_ctx, *out_grad, ref_shape, &outputs);
    } else {
      math::SplitFunctor<DeviceContext, T> split_functor;
      split_functor(dev_ctx, *out_grad, ctx.MultiInput<framework::Tensor>("X"),
                    static_cast<int>(axis), &outputs);
    }
  }
};

}  // namespace operators
}  // namespace paddle
