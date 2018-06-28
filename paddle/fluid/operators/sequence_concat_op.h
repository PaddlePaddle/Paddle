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

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename T>
LoD ConcatLoD(const std::vector<const T*> ins, const size_t level) {
  auto out_lod = ins[0]->lod();
  auto numLevels = ins[0]->NumLevels();
  const size_t n = ins.size();
  const size_t level_idx = ins[0]->NumLevels() - 1 - level;
  for (size_t i = 1; i < n; ++i) {
    for (size_t j = 0; j < ins[i]->lod()[level_idx].size(); ++j) {
      out_lod[level_idx][j] += ins[i]->lod()[level_idx][j];
    }
  }

  for (size_t i = level_idx; i < numLevels - 1; ++i) {
    size_t lod_len = 1;
    for (size_t j = 0; j < n; ++j) {
      lod_len += ins[j]->lod()[i + 1].size() - 1;
    }
    out_lod[i + 1].clear();
    out_lod[i + 1].resize(lod_len);

    size_t idx = 1;
    for (size_t j = 0; j < ins[0]->lod()[i].size() - 1; ++j) {
      for (size_t k = 0; k < n; ++k) {
        for (size_t m = ins[k]->lod()[i][j]; m < ins[k]->lod()[i][j + 1]; ++m) {
          out_lod[i + 1][idx] = out_lod[i + 1][idx - 1] +
                                ins[k]->lod()[i + 1][m + 1] -
                                ins[k]->lod()[i + 1][m];
          idx++;
        }
      }
    }
  }

  return out_lod;
}

template <typename DeviceContext, typename T>
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
    auto out_lod = ins[0]->lod();
    if (axis == 0) {
      out_lod = ConcatLoD<LoDTensor>(ins, level);
    }
    out->set_lod(out_lod);

    const size_t level_idx = out_lod.size() - level - 1;
    auto out_lod_level = framework::ToAbsOffset(out_lod)[level_idx];
    for (size_t i = 0; i < out_lod_level.size() - 1; ++i) {
      Tensor out_t = out->Slice(static_cast<int>(out_lod_level[i]),
                                static_cast<int>(out_lod_level[i + 1]));
      auto out_stride = framework::stride(out_t.dims());
      size_t offset = 0;
      for (size_t j = 0; j < n; ++j) {
        auto in_lod_level = framework::ToAbsOffset(ins[j]->lod())[level_idx];
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

template <typename DeviceContext, typename T>
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
    auto out_lod = ins[0]->lod();
    if (axis == 0UL) {
      out_lod = ConcatLoD<LoDTensor>(ins, level);
    }
    const size_t level_idx = out_lod.size() - level - 1;
    auto out_lod_level = framework::ToAbsOffset(out_lod)[level_idx];

    for (size_t i = 0; i < out_lod_level.size() - 1; ++i) {
      Tensor out_grad_t =
          out_grad->Slice(static_cast<int>(out_lod_level[i]),
                          static_cast<int>(out_lod_level[i + 1]));
      auto out_grad_stride = framework::stride(out_grad_t.dims());
      size_t offset = 0;

      for (size_t j = 0; j < n; ++j) {
        auto x_grad_lod_level =
            framework::ToAbsOffset(x_grads[j]->lod())[level_idx];
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
