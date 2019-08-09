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

class MulOpLite : public OpLite {
 public:
  MulOpLite() {}

  explicit MulOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }
  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    CHECK(!op_desc.Input("X").empty());
    CHECK(!op_desc.Input("Y").empty());
    CHECK(!op_desc.Output("Out").empty());

    auto input = op_desc.Input("X").front();
    auto W = op_desc.Input("Y").front();
    auto out = op_desc.Output("Out").front();
    auto *var = scope->FindVar(input);
    CHECK(var);
    param_.x = &var->Get<Tensor>();
    var = scope->FindVar(W);
    CHECK(var) << "no var called " << W;
    param_.y = &var->Get<Tensor>();
    var = scope->FindVar(out);
    CHECK(var) << "no var called " << out;
    param_.output = var->GetMutable<Tensor>();
    param_.x_num_col_dims = op_desc.GetAttr<int>("x_num_col_dims");
    param_.y_num_col_dims = op_desc.GetAttr<int>("y_num_col_dims");

    return true;
  }

  std::string DebugString() const override { return "mul"; }

 private:
  mutable MulParam param_;
};

#ifdef LITE_WITH_X86
class MulGradOpLite : public OpLite {
 public:
  MulGradOpLite() {}

  explicit MulGradOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override;

  std::string DebugString() const override { return "mul_grad"; }

 private:
  mutable MulGradParam param_;
};
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle
