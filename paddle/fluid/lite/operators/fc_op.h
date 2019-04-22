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
#include "paddle/fluid/lite/core/tensor.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class FcOpLite : public OpLite {
 public:
  FcOpLite() {}

  FcOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShape() const override;

  /*
  bool Run() override {
    CHECK(kernel_);
    kernel_->Run();
    return true;
  }
   */

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const framework::OpDesc &op_desc,
                  lite::Scope *scope) override {
    auto input = op_desc.Input("Input").front();
    auto W = op_desc.Input("W").front();
    auto bias = op_desc.Input("Bias").front();
    auto out = op_desc.Output("Out").front();

    param_.input = scope->FindVar(input)->GetMutable<Tensor>();
    param_.w = scope->FindVar(W)->GetMutable<Tensor>();
    param_.bias = scope->FindVar(bias)->GetMutable<Tensor>();
    CHECK(scope->FindVar(out));
    param_.output = scope->FindVar(out)->GetMutable<Tensor>();
    param_.in_num_col_dims =
        boost::get<int>(op_desc.GetAttr("in_num_col_dims"));

    CHECK(kernel_);
    kernel_->SetParam(param_);

    return true;
  }

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "fc"; }

 private:
  mutable FcParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
