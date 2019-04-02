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

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DGCClipByNormKernel : public ClipByNormKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rampup_begin_step = context.Attr<float>("rampup_begin_step");
    if (static_cast<int>(rampup_begin_step) < 0) {
      return;
    }

    auto current_step_tensor = context.Input<framework::Tensor>("current_step");
    auto* current_step = current_step_tensor->data<T>();

    VLOG(10) << "current_step:" << *current_step
             << ", rampup_begin_step:" << rampup_begin_step;

    if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
      VLOG(10) << "current_step:" << *current_step
               << " < rampup_begin_step:" << rampup_begin_step
               << " so does't use dgc_clip_by_norm";
      return;
    }

    return ClipByNormKernel<DeviceContext, T>::Compute(context);
  };
};

}  // namespace operators
}  // namespace paddle
