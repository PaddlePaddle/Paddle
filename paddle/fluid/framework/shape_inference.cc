/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/shape_inference.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {

std::vector<DDim> InferShapeContext::GetReaderDims(
    const std::string &name) const {
  const std::vector<std::string> &arg_names = Inputs(name);
  PADDLE_ENFORCE_EQ(
      arg_names.size(), 1UL,
      "Reader input '%s' should hold one element, but now it holds %d", name,
      arg_names.size());
  return this->GetRepeatedDims(arg_names[0]);
}

void InferShapeContext::SetReaderDims(const std::string &name,
                                      const std::vector<DDim> &dims) {
  const std::vector<std::string> &arg_names = Outputs(name);
  PADDLE_ENFORCE_EQ(
      arg_names.size(), 1UL,
      "Reader output '%s' should hold one element, but now it holds %d", name,
      arg_names.size());
  return this->SetRepeatedDims(arg_names[0], dims);
}

void InferShapeContext::CheckInputsAndOutputs(const std::string &type) const {
  // 1. Get OpProto
  auto &proto = OpInfoMap::Instance().Get(type).Proto();
  // 2. Check inputs
  for (auto &in : proto.inputs()) {
    if (!in.dispensable()) {
      bool in_exist = false;
      if (!in.duplicable()) {
        in_exist = this->HasInput(in.name());
      } else {
        in_exist = this->HasInputs(in.name());
      }
      PADDLE_ENFORCE_EQ(
          in_exist, true,
          platform::errors::NotFound("No Input(%s) found for %s operator.",
                                     in.name(), type));
    }
  }
  // 3. Check outputs
  for (auto &out : proto.outputs()) {
    if (!out.dispensable()) {
      bool out_exist = false;
      if (!out.duplicable()) {
        out_exist = this->HasOutput(out.name());
      } else {
        out_exist = this->HasOutputs(out.name());
      }
      PADDLE_ENFORCE_EQ(
          out_exist, true,
          platform::errors::NotFound("No Output(%s) found for %s operator.",
                                     out.name(), type));
    }
  }
}

}  // namespace framework
}  // namespace paddle
