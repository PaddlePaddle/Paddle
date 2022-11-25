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

#include "paddle/fluid/operators/clip_by_norm_op.h"
#include "paddle/phi/kernels/clip_by_norm_kernel.h"
#include "paddle/phi/kernels/selected_rows/clip_by_norm_kernel.h"

namespace paddle {
namespace operators {

<<<<<<< HEAD
using Tensor = phi::DenseTensor;
=======
using Tensor = framework::Tensor;
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

template <typename DeviceContext, typename T>
class DGCClipByNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto rampup_begin_step = ctx.Attr<float>("rampup_begin_step");
    if (static_cast<int>(rampup_begin_step) < 0) {
      return;
    }

<<<<<<< HEAD
    auto current_step_tensor = ctx.Input<phi::DenseTensor>("current_step");
=======
    auto current_step_tensor = ctx.Input<framework::Tensor>("current_step");
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
    auto* current_step = current_step_tensor->data<T>();

    VLOG(10) << "current_step:" << *current_step
             << ", rampup_begin_step:" << rampup_begin_step;

    if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
      VLOG(10) << "current_step:" << *current_step
               << " < rampup_begin_step:" << rampup_begin_step
               << " so does't use dgc_clip_by_norm";
      return;
    }

    auto in_var = ctx.InputVar("X");
    auto max_norm = ctx.Attr<float>("max_norm");
    auto& dev_ctx = ctx.device_context<DeviceContext>();

<<<<<<< HEAD
    if (in_var->IsType<phi::DenseTensor>()) {
      auto* x = ctx.Input<phi::DenseTensor>("X");
      auto* y = ctx.Output<phi::DenseTensor>("Out");
=======
    if (in_var->IsType<framework::LoDTensor>()) {
      auto* x = ctx.Input<Tensor>("X");
      auto* y = ctx.Output<Tensor>("Out");
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
      return phi::ClipByNormKernel<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *x,
          max_norm,
          y);
    } else if (in_var->IsType<phi::SelectedRows>()) {
      auto* x = ctx.Input<phi::SelectedRows>("X");
      phi::SelectedRows* output_selected_rows =
          ctx.Output<phi::SelectedRows>("Out");
      return phi::sr::ClipByNormKernel<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *x,
          max_norm,
          output_selected_rows);
    }
  };
};

}  // namespace operators
}  // namespace paddle
