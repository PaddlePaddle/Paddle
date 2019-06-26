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

class FcOpLite : public OpLite {
 public:
  FcOpLite() {}

  explicit FcOpLite(const std::string &type) : OpLite(type) {}

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
  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override {
    auto input = op_desc.Input("Input").front();
    auto W = op_desc.Input("W").front();
    auto bias = op_desc.Input("Bias").front();
    auto out = op_desc.Output("Out").front();

    param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
    param_.w = scope->FindVar(W)->GetMutable<lite::Tensor>();
    param_.bias = scope->FindVar(bias)->GetMutable<lite::Tensor>();
    CHECK(scope->FindVar(out));
    param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();
    param_.in_num_col_dims = op_desc.GetAttr<int>("in_num_col_dims");

    // For Int8
    if (op_desc.HasAttr("enable_int8")) {
      param_.enable_int8 = op_desc.GetAttr<bool>("enable_int8");
      if (op_desc.HasAttr("input_scale"))
        param_.input_scale = op_desc.GetAttr<float>("input_scale");
      if (op_desc.HasAttr("weight_scale"))
        param_.weight_scale =
            op_desc.GetAttr<std::vector<float>>("weight_scale");
      if (op_desc.HasAttr("output_scale"))
        param_.output_scale = op_desc.GetAttr<float>("output_scale");
    }
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
