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
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

class DropoutOpLite : public OpLite {
 public:
  explicit DropoutOpLite(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.x);
    return true;
  }

  bool InferShape() const override {
    const auto x_dims = param_.x->dims();
    param_.output->Resize(x_dims);
    if (param_.is_test == false) {
      param_.mask->Resize(x_dims);
    }
    // share LoD
    // param_.output->set_lod(param_.input->lod());
    return true;
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }
  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) override {
    auto input = op_desc.Input("X").front();
    auto out = op_desc.Output("Out").front();
    auto Mask = op_desc.Output("Mask").front();

    param_.x = GetVar<lite::Tensor>(scope, input);
    param_.output = GetMutableVar<lite::Tensor>(scope, out);
    param_.mask = GetMutableVar<lite::Tensor>(scope, Mask);

    param_.dropout_prob = op_desc.GetAttr<float>("dropout_prob");
    param_.is_test = true;
    // TODO(sangoly): `is_test` has different attr type in x86 and arm, set
    // `true` now.
    // if (op_desc.HasAttr("is_test")) {
    //   param_.is_test = op_desc.GetAttr<bool>("is_test");
    // }
    param_.fix_seed = op_desc.GetAttr<bool>("fix_seed");
    param_.seed = op_desc.GetAttr<int>("seed");
    param_.dropout_implementation =
        op_desc.GetAttr<std::string>("dropout_implementation");
    return true;
  }

  std::string DebugString() const override { return "dropout"; }

 private:
  mutable DropoutParam param_;
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(dropout, paddle::lite::operators::DropoutOpLite);
