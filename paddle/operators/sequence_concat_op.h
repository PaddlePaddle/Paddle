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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename T>
LoD concatLoD(const std::vector<const T*> ins, const size_t axis,
              const size_t level) {
  auto out_lod = ins[0]->lod();
  const size_t n = ins.size();
  if (axis == 0UL) {
    for (size_t i = 1; i < n; ++i) {
      for (size_t j = 0; j < ins[i]->lod()[0].size(); ++j) {
        out_lod[0][j] += ins[i]->lod()[0][j];
      }

      if (ins[0]->NumLevels() == 2) {
        for (size_t j = 1; j < ins[i]->lod()[1].size(); ++j) {
          if (level == 0UL) {
            out_lod[1].push_back(out_lod[1].back() + ins[i]->lod()[1][j] -
                                 ins[i]->lod()[1][j - 1]);
          } else if (level == 1UL) {
            out_lod[1][j] += ins[1]->lod()[1][j];
          }
        }
      }
    }
  }
  return out_lod;
}

template <typename Place, typename T>
class SequenceConcatOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    const size_t axis = static_cast<size_t>(ctx.Attr<int>("axis"));
    const size_t level = static_cast<size_t>(ctx.Attr<int>("level"));
    const size_t n = ins.size();

    for (size_t i = 1; i < n; ++i) {
      PADDLE_ENFORCE_EQ(ins[0]->NumLevels(), ins[i]->NumLevels(),
                        "The levels of all the input LoDTensors "
                        "should be the same.");
      PADDLE_ENFORCE_EQ(ins[0]->dims().size(), ins[i]->dims().size(),
                        "The dimension size of all the input LoDTensors "
                        "should be the same.");

      const size_t dims_size = ins[i]->dims().size();
      for (size_t j = 0; j < dims_size; ++j) {
        if (j == axis) continue;
        PADDLE_ENFORCE_EQ(ins[0]->dims()[j], ins[i]->dims()[j],
                          "Except for the dimension of the specified "
                          "axis along which all the inputs are concatenated, "
                          "dimensions of all the other axises of the input "
                          "LoDTensors should be the same.");
      }
    }
    PADDLE_ENFORCE_GT(ins[0]->NumLevels(), level,
                      "The levels of all the input LoDTensors "
                      "should be greater than the specify level");

    out->mutable_data<T>(ctx.GetPlace());
    auto out_lod = concatLoD<LoDTensor>(ins, axis, level);
    out->set_lod(out_lod);

    auto out_lod_level = out_lod[level];
    for (size_t i = 0; i < out_lod_level.size() - 1; ++i) {
      Tensor out_t = out->Slice(static_cast<int>(out_lod_level[i]),
                                static_cast<int>(out_lod_level[i + 1]));
      auto out_stride = framework::stride(out_t.dims());
      size_t offset = 0;

      for (size_t j = 0; j < n; ++j) {
        auto in_lod_level = ins[j]->lod()[level];
        auto in_stride = framework::stride(ins[j]->dims());
        Tensor in_t = ins[j]->Slice(static_cast<int>(in_lod_level[i]),
                                    static_cast<int>(in_lod_level[i + 1]));
        size_t axis_dim = in_t.dims()[axis];
        StridedMemcpy<T>(ctx.device_context(), in_t.data<T>(), in_stride,
                         in_t.dims(), out_stride, out_t.data<T>() + offset);
        offset += axis_dim * in_stride[axis];
      }
    }
  }
};

template <typename Place, typename T>
class SequenceConcatGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    auto* out_grad =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto x_grads =
        ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));
    size_t axis = static_cast<size_t>(ctx.Attr<int>("axis"));
    size_t level = static_cast<size_t>(ctx.Attr<int>("level"));
    const size_t n = x_grads.size();

    // Set Grad(X) LoD as X
    for (size_t i = 0; i < n; i++) {
      x_grads[i]->set_lod(ins[i]->lod());
      x_grads[i]->mutable_data<T>(ctx.GetPlace());
    }

    auto out_lod = concatLoD<LoDTensor>(ins, axis, level);
    auto out_lod_level = out_lod[level];

    for (size_t i = 0; i < out_lod_level.size() - 1; ++i) {
      Tensor out_grad_t =
          out_grad->Slice(static_cast<int>(out_lod_level[i]),
                          static_cast<int>(out_lod_level[i + 1]));
      auto out_grad_stride = framework::stride(out_grad_t.dims());
      size_t offset = 0;

      for (size_t j = 0; j < n; ++j) {
        auto x_grad_lod_level = x_grads[j]->lod()[level];
        auto x_grad_stride = framework::stride(x_grads[j]->dims());
        Tensor x_grad_t =
            x_grads[j]->Slice(static_cast<int>(x_grad_lod_level[i]),
                              static_cast<int>(x_grad_lod_level[i + 1]));
        size_t axis_dim = x_grad_t.dims()[axis];
        StridedMemcpy<T>(ctx.device_context(), out_grad_t.data<T>() + offset,
                         out_grad_stride, out_grad_t.dims(), x_grad_stride,
                         x_grad_t.data<T>());
        offset += axis_dim * out_grad_stride[axis];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
