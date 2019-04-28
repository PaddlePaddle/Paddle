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

inline static framework::DDim GetOutputDim(
    const std::vector<framework::DDim>& ins, size_t axis, bool is_runtime) {
  const size_t n = ins.size();

  PADDLE_ENFORCE_GT(n, 0, "Input tensors count should > 0.");
  if (n == 1) {
    VLOG(3) << "Warning: concat op have only one input, may waste memory";
  }

  auto out_dims = ins[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        if (is_runtime) {
          out_dims[axis] += ins[i][j];
        } else {
          if (ins[i][j] == -1) {
            out_dims[axis] = -1;
          } else {
            out_dims[axis] += ins[i][j];
          }
        }
      } else {
        if (is_runtime) {
          // check all shape in run time
          PADDLE_ENFORCE_EQ(out_dims[j], ins[i][j],
                            "Input tensors should have the same "
                            "elements except the specify axis.");
        } else {
          // not check -1 with other in compile time
          if (out_dims[j] > 0 && ins[i][j] > 0) {
            PADDLE_ENFORCE_EQ(out_dims[j], ins[i][j],
                              "Input tensors should have the same "
                              "elements except the specify axis.");
          }
        }
      }
    }
  }
  if (out_dims[axis] < 0) {
    out_dims[axis] = -1;
  }

  return out_dims;
}

template <typename DeviceContext, typename T>
class ConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    framework::LoDTensor* out = ctx.Output<framework::LoDTensor>("Out");
    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));
    auto place = ctx.GetPlace();

    PADDLE_ENFORCE_NOT_NULL(out);

    // Compute the out's dims and share x's lod
    std::vector<framework::DDim> in_dims;
    for (size_t i = 0; i < ins.size(); ++i) {
      in_dims.push_back(ins[i]->dims());
    }
    framework::DDim out_dims =
        GetOutputDim(in_dims, axis, /* is_runtime */ true);
    out->mutable_data<T>(out_dims, place);
    out->set_lod(ins[0]->lod());

    // Sometimes direct copies will be faster, this maybe need deeply analysis.
    if (axis == 0 && ins.size() < 10) {
      size_t output_offset = 0;
      for (auto* in : ins) {
        auto in_stride = framework::stride_numel(in->dims());
        auto out_stride = framework::stride_numel(out->dims());
        StridedNumelCopyWithAxis<T>(ctx.device_context(), axis,
                                    out->data<T>() + output_offset, out_stride,
                                    in->data<T>(), in_stride, in_stride[axis]);
        output_offset += in_stride[axis];
      }
    } else {
      std::vector<framework::Tensor> inputs(ins.size());
      for (size_t j = 0; j < ins.size(); ++j) {
        inputs[j] = *ins[j];
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
    auto ins_grad =
        ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));

    // Share ins' lod to ins_grad
    for (size_t i = 0; i < ins_grad.size(); ++i) {
      if (ins_grad[i] != nullptr) {
        ins_grad[i]->set_lod(ins[i]->lod());
      }
    }

    int64_t axis = static_cast<int64_t>(ctx.Attr<int>("axis"));

    // get output tensor that the name is not kEmptyVarName
    std::vector<framework::Tensor*> outputs;
    for (size_t j = 0; j < ins_grad.size(); ++j) {
      if (out_var_names[j] != framework::kEmptyVarName) {
        // Share ins' dims to ins_grad
        ins_grad[j]->mutable_data<T>(ins[j]->dims(), ctx.GetPlace());
        outputs.push_back(ins_grad[j]);
      } else {
        outputs.push_back(nullptr);
      }
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // Sometimes direct copies will be faster, this maybe need deeply analysis.
    if (axis == 0 && ins_grad.size() < 10) {
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
