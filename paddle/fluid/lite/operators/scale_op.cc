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

#include <string>
#include <vector>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/core/tensor.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class ScaleOp : public OpLite {
 public:
  ScaleOp() {}

  explicit ScaleOp(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.x);
    CHECK_OR_FALSE(param_.output);
    return true;
  }

  bool InferShape() const override {
    param_.output->Resize(param_.x->dims());
    return true;
  }

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const OpDesc &op_desc, lite::Scope *scope) override {
    auto x = op_desc.Input("X").front();
    auto out = op_desc.Output("Out").front();

    param_.x = scope->FindVar(x)->GetMutable<Tensor>();
    CHECK(scope->FindVar(out));
    param_.output = scope->FindVar(out)->GetMutable<Tensor>();
    param_.scale = op_desc.GetAttr("scale").get<float>();
    param_.bias = op_desc.GetAttr("bias").get<float>();
    param_.bias_after_scale = op_desc.GetAttr("bias_after_scale").get<bool>();
    return true;
  }

  std::string DebugString() const override { return op_type_; }

 private:
  mutable ScaleParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(scale, paddle::lite::operators::ScaleOp);
