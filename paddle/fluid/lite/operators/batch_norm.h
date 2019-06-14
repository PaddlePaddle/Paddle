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
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class BatchNormOpLite : public OpLite {
 public:
  BatchNormOpLite() {}

  explicit BatchNormOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    auto x = op_desc.Input("X").front();
    auto bias = op_desc.Input("Bias").front();
    auto mean = op_desc.Input("Mean").front();
    auto scale = op_desc.Input("Scale").front();
    auto variance = op_desc.Input("Variance").front();

    auto out = op_desc.Output("Y").front();
    auto mean_out = op_desc.Output("MeanOut").front();
    auto var_out = op_desc.Output("VarianceOut").front();
    auto saved_mean = op_desc.Output("SavedMean").front();
    auto saved_var = op_desc.Output("SavedVariance").front();

    auto *var = scope->FindVar(x);
    param_.x = var->GetMutable<Tensor>();
    var = scope->FindVar(bias);
    param_.bias = var->GetMutable<Tensor>();
    var = scope->FindVar(mean);
    param_.mean = var->GetMutable<Tensor>();
    var = scope->FindVar(scale);
    param_.scale = var->GetMutable<Tensor>();
    var = scope->FindVar(variance);
    param_.var = var->GetMutable<Tensor>();
    var = scope->FindVar(out);
    param_.out = var->GetMutable<Tensor>();
    var = scope->FindVar(mean_out);
    param_.mean_out = var->GetMutable<Tensor>();
    var = scope->FindVar(var_out);
    param_.var_out = var->GetMutable<Tensor>();
    var = scope->FindVar(saved_mean);
    param_.saved_mean = var->GetMutable<Tensor>();
    var = scope->FindVar(saved_var);
    param_.saved_var = var->GetMutable<Tensor>();

    param_.eps = op_desc.GetAttr<float>("epsilon");

    return true;
  }

  std::string DebugString() const override { return "batch_norm"; }

 private:
  mutable BatchNormParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
