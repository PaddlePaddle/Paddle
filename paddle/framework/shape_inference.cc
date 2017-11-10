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
#include "paddle/framework/shape_inference.h"

namespace paddle {
namespace framework {

std::vector<framework::DDim> InferShapeContext::GetInputsDim(
    const std::string &name) const {
  const std::vector<std::string> &names = Inputs(name);
  return GetDims(names);
}

void InferShapeContext::SetOutputsDim(
    const std::string &name, const std::vector<framework::DDim> &dims) {
  auto &names = Outputs(name);
  SetDims(names, dims);
}

std::vector<framework::DDim> InferShapeContext::GetDims(
    const std::vector<std::string> &names) const {
  std::vector<framework::DDim> ret;
  ret.reserve(names.size());
  std::transform(
      names.begin(), names.end(), std::back_inserter(ret),
      [this](const std::string &name) { return this->GetDim(name); });
  return ret;
}

void InferShapeContext::SetDims(const std::vector<std::string> &names,
                                const std::vector<framework::DDim> &dims) {
  size_t length = names.size();
  PADDLE_ENFORCE_EQ(length, dims.size());
  for (size_t i = 0; i < length; ++i) {
    SetDim(names[i], dims[i]);
  }
}
std::vector<VarDesc::VarType> InferShapeContext::GetInputsVarType(
    const std::string &name) const {
  return GetVarTypes(Inputs(name));
}
std::vector<VarDesc::VarType> InferShapeContext::GetOutputsVarType(
    const std::string &name) const {
  return GetVarTypes(Outputs(name));
}
std::vector<VarDesc::VarType> InferShapeContext::GetVarTypes(
    const std::vector<std::string> &names) const {
  std::vector<VarDesc::VarType> retv;
  retv.resize(names.size());
  std::transform(names.begin(), names.end(), retv.begin(),
                 std::bind(std::mem_fn(&InferShapeContext::GetVarType), this,
                           std::placeholders::_1));
  return retv;
}

}  // namespace framework
}  // namespace paddle
