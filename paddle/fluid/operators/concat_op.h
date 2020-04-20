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

#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {
static inline framework::DDim ComputeAndCheckShape(
    const bool is_runtime, const std::vector<framework::DDim>& inputs_dims,
    const size_t axis) {
  const size_t n = inputs_dims.size();
  auto out_dims = inputs_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        if (is_runtime) {
          out_dims[axis] += inputs_dims[i][j];
        } else {
          if (inputs_dims[i][j] == -1) {
            out_dims[axis] = -1;
          } else {
            out_dims[axis] += inputs_dims[i][j];
          }
        }
      } else {
        bool check_shape =
            is_runtime || (out_dims[j] > 0 && inputs_dims[i][j] > 0);
        if (check_shape) {
          // check all shape in run time
          PADDLE_ENFORCE_EQ(inputs_dims[0][j], inputs_dims[i][j],
                            platform::errors::InvalidArgument(
                                "The %d-th dimension of input[0] and input[%d] "
                                "is expected to be equal."
                                "But received input[0]'s shape = "
                                "[%s], input[%d]'s shape = [%s].",
                                j, i, inputs_dims[0], i, inputs_dims[i]));
        }
      }
    }
  }
  return out_dims;
}

static inline int64_t ComputeAxis(int64_t axis, int64_t rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank, true,
      platform::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d", -rank,
          rank, axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

template <typename DeviceContext, typename T>
class ConcatKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    framework::LoDTensor* out = ctx.Output<framework::LoDTensor>("Out");
    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));
    auto axis = ctx.Attr<int>("axis");
    bool need_resize_out_dims = false;
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<framework::Tensor>("AxisTensor");
      axis = GetDataFromTensor<int>(axis_tensor)[0];
      need_resize_out_dims = true;
    }
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));

    if (need_resize_out_dims) {
      const size_t n = ins.size();
      std::vector<framework::DDim> ins_dims(n);
      for (size_t i = 0; i < n; i++) {
        ins_dims[i] = ins[i]->dims();
      }

      framework::DDim out_dims = ComputeAndCheckShape(true, ins_dims, axis);
      out->Resize(out_dims);
    }
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    // If axis is 0, the lod of the output is not the same as inputs.
    if (axis == 0 && ins[0]->lod().size() > 0) {
      size_t lod_size_0 = ins[0]->lod().size();
      size_t lod_size = lod_size_0;
      for (size_t i = 1; i < ins.size(); ++i) {
        if (ins[i]->lod().size() > 0) {
          PADDLE_ENFORCE_EQ(
              ins[i]->lod().size(), lod_size_0,
              platform::errors::Unimplemented(
                  "The lod level of all input LoDTensors should be same. "
                  "Maybe different lod level of input LoDTensors can concat,"
                  "it is not supported currently. The lod level of %dth input "
                  "is %d and first input is %d.",
                  i, ins[i]->lod().size(), lod_size_0));
        } else {
          lod_size = 0;
          break;
        }
      }
      if (lod_size) {
        auto* out_lod = out->mutable_lod();
        for (size_t i = 1; i < ins.size(); ++i) {
          auto in_lod = ConvertToLengthBasedLoD(ins[i]->lod());
          AppendLoD(out_lod, in_lod);
        }
      }
    }

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
    auto out_var_names = ctx.OutputNames(framework::GradVarName("X"));
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
    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));

    auto axis = ctx.Attr<int>("axis");
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<framework::Tensor>("AxisTensor");
      axis = GetDataFromTensor<int>(axis_tensor)[0];
    }
    axis = ComputeAxis(static_cast<int64_t>(axis),
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
