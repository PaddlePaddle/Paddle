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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence_padding.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename DeviceContext, typename T>
class SequenceUnpadOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x_t = ctx.Input<LoDTensor>("X");
    auto* len_t = ctx.Input<LoDTensor>("Length");
    auto* out_t = ctx.Output<LoDTensor>("Out");
    out_t->mutable_data<T>(ctx.GetPlace());

    const int64_t* seq_len_ptr = nullptr;
    if (platform::is_gpu_place(ctx.GetPlace())) {
      LoDTensor seq_len_cpu;
      seq_len_cpu.Resize(len_t->dims());
      seq_len_ptr = seq_len_cpu.mutable_data<int64_t>(platform::CPUPlace());
      framework::TensorCopy(*len_t, platform::CPUPlace(),
                            ctx.template device_context<DeviceContext>(),
                            &seq_len_cpu);
    } else {
      seq_len_ptr = len_t->data<int64_t>();
    }

    size_t batch_size = x_t->dims()[0];
    std::vector<size_t> out_lod0(batch_size + 1, 0);
    for (size_t i = 0; i < batch_size; ++i) {
      out_lod0[i + 1] = out_lod0[i] + seq_len_ptr[i];
    }

    framework::LoD out_lod;
    out_lod.push_back(out_lod0);
    out_t->set_lod(out_lod);

    std::vector<int64_t> out_dims_vec{static_cast<int64_t>(out_lod0.back())};
    if (x_t->dims().size() == 2) {
      out_dims_vec.push_back(1);
    } else {
      for (int i = 2; i < x_t->dims().size(); ++i) {
        out_dims_vec.push_back(x_t->dims()[i]);
      }
    }
    out_t->Resize(framework::make_ddim(out_dims_vec));

    int64_t padded_length = x_t->dims()[1];
    math::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), *x_t, out_t,
        padded_length, 0, false, math::kBatchLengthWidth);
  }
};

template <typename DeviceContext, typename T>
class SequenceUnpadGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    if (d_x) {
      const auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
      d_x->mutable_data<T>(ctx.GetPlace());

      int padded_length = d_x->dims()[1];

      LoDTensor zero_pads;
      zero_pads.Resize({1, 1});
      zero_pads.mutable_data<T>(ctx.GetPlace());
      math::SetConstant<DeviceContext, T> set_zero;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      set_zero(dev_ctx, &zero_pads, static_cast<T>(0));

      math::PaddingLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), *d_out, d_x, zero_pads,
          padded_length, 0, false, math::kBatchLengthWidth);
    }
  }
};

}  // namespace operators
}  // namespace paddle
