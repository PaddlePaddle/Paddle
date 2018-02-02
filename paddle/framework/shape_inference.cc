/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "grad_op_desc_maker.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {

DDim InferShapeContext::GetInputDim(const std::string &name) const {
  const std::vector<std::string> &arg_names = Inputs(name);
  PADDLE_ENFORCE_EQ(arg_names.size(), 1UL,
                    "Input(%s) should hold one element, but now it holds %d",
                    name, arg_names.size());
  return this->GetDim(arg_names[0]);
}

std::vector<DDim> InferShapeContext::GetInputsDim(
    const std::string &name) const {
  const std::vector<std::string> &arg_names = Inputs(name);
  return GetDims(arg_names);
}

DDim InferShapeContext::GetInputsElementDim(const std::string &name,
                                            int idx) const {
  const std::vector<std::string> &names = Inputs(name);
  return this->GetDim(names[idx]);
}

void InferShapeContext::SetOutputDim(const std::string &name, const DDim &dim) {
  auto &arg_names = Outputs(name);
  PADDLE_ENFORCE_EQ(arg_names.size(), 1UL,
                    "Output(%s) should hold one element, but now it holds %d",
                    name, arg_names.size());
  SetDim(arg_names[0], dim);
}

void InferShapeContext::SetOutputsDim(const std::string &name,
                                      const std::vector<DDim> &dims) {
  auto &names = Outputs(name);
  SetDims(names, dims);
}

std::vector<DDim> InferShapeContext::GetDims(
    const std::vector<std::string> &names) const {
  std::vector<DDim> ret;
  ret.reserve(names.size());
  std::transform(
      names.begin(), names.end(), std::back_inserter(ret),
      [this](const std::string &name) { return this->GetDim(name); });
  return ret;
}
void InferShapeContext::SetDims(const std::vector<std::string> &names,
                                const std::vector<DDim> &dims) {
  size_t length = names.size();
  PADDLE_ENFORCE_EQ(length, dims.size());
  for (size_t i = 0; i < length; ++i) {
    if (names[i] == framework::kEmptyVarName) {
      continue;
    }
    SetDim(names[i], dims[i]);
  }
}
std::vector<proto::VarDesc::VarType> InferShapeContext::GetInputsVarType(
    const std::string &name) const {
  return GetVarTypes(Inputs(name));
}
std::vector<proto::VarDesc::VarType> InferShapeContext::GetOutputsVarType(
    const std::string &name) const {
  return GetVarTypes(Outputs(name));
}
std::vector<proto::VarDesc::VarType> InferShapeContext::GetVarTypes(
    const std::vector<std::string> &names) const {
  std::vector<proto::VarDesc::VarType> retv;
  retv.resize(names.size());
  std::transform(names.begin(), names.end(), retv.begin(),
                 std::bind(std::mem_fn(&InferShapeContext::GetVarType), this,
                           std::placeholders::_1));
  return retv;
}

}  // namespace framework
}  // namespace paddle
