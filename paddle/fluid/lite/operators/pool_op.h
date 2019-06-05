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

class PoolOpLite : public OpLite {
 public:
  PoolOpLite() {}

  explicit PoolOpLite(const std::string &type) : OpLite(type) {}

  int PoolOutputSize(int input_size, int filter_size, int padding, int stride,
                     bool ceil_mode);

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
  bool AttachImpl(const OpDesc &op_desc, lite::Scope *scope) override {
    auto x = op_desc.Input("X").front();
    auto out = op_desc.Output("Out").front();

    CHECK(scope->FindVar(x));
    CHECK(scope->FindVar(out));
    param_.x = scope->FindVar(x)->GetMutable<lite::Tensor>();
    param_.out = scope->FindVar(out)->GetMutable<lite::Tensor>();

    param_.pooling_type = GetAttr<std::string>(op_desc.GetAttr("pooling_type"));
    param_.ksize = GetAttr<std::vector<int>>(op_desc.GetAttr("ksize"));
    param_.global_pooling = GetAttr<bool>(op_desc.GetAttr("global_pooling"));
    param_.strides = GetAttr<std::vector<int>>(op_desc.GetAttr("strides"));
    param_.paddings = GetAttr<std::vector<int>>(op_desc.GetAttr("paddings"));

    param_.exclusive = GetAttr<bool>(op_desc.GetAttr("exclusive"));
    param_.adaptive = GetAttr<bool>(op_desc.GetAttr("adaptive"));
    param_.ceil_mode = GetAttr<bool>(op_desc.GetAttr("ceil_mode"));
    param_.use_quantizer = GetAttr<bool>(op_desc.GetAttr("use_quantizer"));
    param_.data_format = GetAttr<std::string>(op_desc.GetAttr("data_format"));
    return true;
  }

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "pool"; }

 private:
  mutable PoolParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
