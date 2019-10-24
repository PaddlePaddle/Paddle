// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>

#include "paddle/fluid/operators/optimizers/momentum_op.h"
#include "paddle/fluid/operators/optimizers/sgd_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DGCMomentumKernel : public framework::OpKernel<T> {
 public:
  DGCMomentumKernel()
      : _momentum_op_kernel(new MomentumOpKernel<DeviceContext, T>()),
        _sgd_op_kernel(new SGDOpKernel<DeviceContext, T>()) {}

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
      VLOG(10) << " so use momentum optimizer";
      return _momentum_op_kernel->Compute(context);
    }

    VLOG(10) << " so use sgd optimizer";
    return _sgd_op_kernel->Compute(context);
  }

 private:
  std::unique_ptr<MomentumOpKernel<DeviceContext, T>> _momentum_op_kernel;
  std::unique_ptr<SGDOpKernel<DeviceContext, T>> _sgd_op_kernel;
};

}  // namespace operators
}  // namespace paddle
