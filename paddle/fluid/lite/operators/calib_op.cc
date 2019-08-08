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

#include "paddle/fluid/lite/operators/calib_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CalibOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  return true;
}
bool CalibOpLite::InferShape() const {
  param_.output->Resize(param_.input->dims());
  return true;
}

bool CalibOpLite::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto x_var = scope->FindVar(opdesc.Input("Input").front());
  auto output_var = scope->FindVar(opdesc.Output("Out").front());
  CHECK(x_var);
  CHECK(output_var);
  param_.input = const_cast<lite::Tensor *>(&(x_var->Get<lite::Tensor>()));
  param_.output = output_var->GetMutable<lite::Tensor>();
  std::vector<std::string> input_arg_names = opdesc.InputArgumentNames();
  if (opdesc.HasAttr("scale")) {
    param_.scale = opdesc.GetAttr<float>("scale");
  }
  CHECK(param_.input) << "Input(X) of CalibOp should not be null.";
  CHECK(param_.output) << "Output(Out) of CalibOp should not be null.";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(calib, paddle::lite::operators::CalibOpLite);
