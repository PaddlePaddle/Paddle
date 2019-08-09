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
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class FakeDequantizeMaxAbsOpLite : public OpLite {
 public:
  FakeDequantizeMaxAbsOpLite() {}

  explicit FakeDequantizeMaxAbsOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override { return true; }

  bool InferShape() const override { return true; }

  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    auto x = op_desc.Input("X").front();
    auto in_scale = op_desc.Input("Scale").front();

    auto out = op_desc.Output("Out").front();

    param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
    param_.in_scale = scope->FindVar(in_scale)->GetMutable<lite::Tensor>();

    param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();
    param_.max_range = op_desc.GetAttr<float>("max_range");
    return true;
  }

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "fake_dequantize_max_abs"; }

 private:
  mutable FakeDequantizeMaxAbsParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
