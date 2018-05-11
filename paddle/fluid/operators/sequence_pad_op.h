/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence_padding.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename DeviceContext, typename T>
struct CopyFunctor {
  LoDTensor* lod_tensor_;
  LoDTensor* pad_tensor_;
  const LoD& ref_lod_;
  const DeviceContext& ctx_;
  bool is_lod_to_pad_;

  CopyFunctor(LoDTensor* lod_tensor, const LoD& ref_lod, LoDTensor* pad_tensor,
              const DeviceContext& ctx, bool is_lod_to_pad)
      : lod_tensor_(lod_tensor),
        pad_tensor_(pad_tensor),
        ref_lod_(ref_lod),
        ctx_(ctx),
        is_lod_to_pad_(is_lod_to_pad) {}

  void operator()() const {
    /*
    auto seq_num = ref_lod_.size() - 1;
    auto max_len = pad_tensor_->dims()[0] / seq_num;

    PADDLE_ENFORCE_EQ(max_len * seq_num, pad_tensor_->dims()[0],
                      "First dimension of padded tensor should be equal to "
                      "maximum sequence length mulplied by sequence number.");

    for (size_t i = 1; i < ref_lod_.size(); ++i) {
      auto seq_start = ref_lod_[i - 1];
      auto seq_end = ref_lod_[i];
      auto pad_start = (i - 1) * max_len;
      auto pad_end = pad_start + (seq_end - seq_start);
      auto sub_lod_tensor = lod_tensor_->Slice(seq_start, seq_end);
      auto sub_pad_tensor = pad_tensor_->Slice(pad_start, pad_end);
      if (is_lod_to_pad_) {
        framework::TensorCopy(sub_lod_tensor, ctx.GetPlace(), &sub_pad_tensor);
      } else {
        framework::TensorCopy(sub_pad_tensor, ctx.GetPlace(), &sub_lod_tensor);
      }
    }
    */
  }
};

template <typename DeviceContext, typename T>
class SequencePadOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    /*
    auto* x = ctx.Input<LoDTensor>("X");
    auto* out_ptr = ctx.Output<LoDTensor>("Out");

    out_ptr->mutable_data<T>(ctx.GetPlace());

    // Resize();

    T pad_value = static_cast<T>(ctx.Attr<float>("pad_value"));

    math::PaddingLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), *x, *, false);

    math::SetConstant<DeviceContext, T> set_func;
    set_func(ctx.template device_context<DeviceContext>(), out_ptr, pad_value);
    */
  }
};

template <typename DeviceContext, typename T>
class SequencePadGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    /*
    auto* x_ptr = ctx.Input<LoDTensor>("X");
    auto* g_out_ptr = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* g_x_ptr = ctx.Output<LoDTensor>(framework::GradVarName("X"));

    math::SetConstant<DeviceContext, T> set_func;
    set_func(ctx.template device_context<DeviceContext>(),
             g_x_ptr,
             static_cast<T>(0));

    auto& x_lod = x_ptr->lod();
    auto& x_last_level_lod = x_lod[x_lod.size() - 1];

    CopyFunctor copy_func<DeviceContext, T>(g_out_ptr,
                                            x_last_level_lod,
                                            g_x_ptr,
                                            ctx,
                                            false);
    copy_func();
    */
  }
};

}  // namespace operators
}  // namespace paddle
