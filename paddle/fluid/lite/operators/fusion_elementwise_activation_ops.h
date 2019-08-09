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
#include <string>
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/operators/elementwise_ops.h"

namespace paddle {
namespace lite {
namespace operators {

class FusionElementwiseActivationOp : public OpLite {
 public:
  explicit FusionElementwiseActivationOp(const std::string& type)
      : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override;

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override {
    return "fusion_elementwise_activation_op";
  }

 private:
  mutable operators::FusionElementwiseActivationParam param_;
};

#ifdef LITE_WITH_X86
class FusionElementwiseActivationGradExplicitOp : public OpLite {
 public:
  explicit FusionElementwiseActivationGradExplicitOp(const std::string& type)
      : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override;

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override {
    return "fusion_elementwise_activation_grad_explicit_op";
  }

 private:
  mutable operators::FusionElementwiseActivationGradParam param_;
};
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle
