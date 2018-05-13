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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {
namespace operators {

class TensorRTEngineOp : public framework::OperatorWithKernel {
 public:
  TensorRTEngineOp() = default;
};

template <typename DeviceContext, typename T>
class TensorRTEngineKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    if (!engine_) {
      Prepare(context);
    }
    auto& inputs = context.Inputs("Xs");
    PADDLE_ENFORCE(!inputs.empty(), "should pass more than one inputs");
    auto* var0 = context.Input(inputs.front());
    PADDLE_ENFORCE_NOT_NULL(var0);
    auto* tensor0 = var0->GetMutable<framework::LoDTensor>();
    const batch_size = tensor0->dims()[0];

    // Convert input tensor from fluid to engine.
    for (const auto& x : context.Inputs("Xs")) {
      // convert input and copy to TRT engine's buffer
    }
    // Execute the engine.
    PADDLE_ENFORCE_GT(max_batch_, 0);
    engine_->Execute(max_batch_);
    // Convert output tensor from engine to fluid
    for (const auto& y : context.Outputs("Ys")) {
      // convert output and copy to fluid.
    }
  }

 protected:
  // Build the engine.
  void Prepare(const framework::ExecutionContext& context) const;

 private:
  mutable std::unique_ptr<inference::tensorrt::TensorRTEngine> engine_;
  mutable int max_batch_{0};
};

}  // namespace operators
}  // namespace paddle
