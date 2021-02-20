/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/extension/include/op_meta_info.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/custom_operator.h"

namespace paddle {

////////////////////// Op Meta Info //////////////////////

OpMetaInfo& OpMetaInfo::Inputs(std::vector<std::string>&& inputs) {
  inputs_ = std::forward<std::vector<std::string>>(inputs);
  return *this;
}
OpMetaInfo& OpMetaInfo::Outputs(std::vector<std::string>&& outputs) {
  outputs_ = std::forward<std::vector<std::string>>(outputs);
  return *this;
}
OpMetaInfo& OpMetaInfo::SetKernelFn(KernelFunc&& func) {
  kernel_fn_ = std::forward<KernelFunc>(func);
  return *this;
}
OpMetaInfo& OpMetaInfo::SetInferShapeFn(InferShapeFunc&& func) {
  infer_shape_fn_ = std::forward<InferShapeFunc>(func);
  return *this;
}
OpMetaInfo& OpMetaInfo::SetInferDtypeFn(InferDtypeFunc&& func) {
  infer_dtype_fn_ = std::forward<InferDtypeFunc>(func);
  return *this;
}

//////////////// Op Meta Info Map /////////////////

std::vector<OpMetaInfo>& OpMetaInfoMap::operator[](const std::string& name) {
  return map_[name];
}

const std::unordered_map<std::string, std::vector<OpMetaInfo>>&
OpMetaInfoMap::GetMap() const {
  return map_;
}

//////////////// Op Meta Info Builder /////////////////

OpMetaInfoBuilder::OpMetaInfoBuilder(std::string&& name) {
  name_ = std::forward<std::string>(name);
  auto& info_vector = OpMetaInfoMap::Instance()[name_];
  auto op_meta = OpMetaInfo(name_);
  info_vector.emplace_back(std::move(op_meta));
  info_ptr_ = &(info_vector.back());
}

OpMetaInfoBuilder& OpMetaInfoBuilder::Inputs(
    std::vector<std::string>&& inputs) {
  info_ptr_->Inputs(std::forward<std::vector<std::string>>(inputs));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::Outputs(
    std::vector<std::string>&& outputs) {
  info_ptr_->Outputs(std::forward<std::vector<std::string>>(outputs));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetKernelFn(KernelFunc&& func) {
  info_ptr_->SetKernelFn(std::forward<KernelFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInferShapeFn(InferShapeFunc&& func) {
  info_ptr_->SetInferShapeFn(std::forward<InferShapeFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInferDtypeFn(InferDtypeFunc&& func) {
  info_ptr_->SetInferDtypeFn(std::forward<InferDtypeFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetBackwardOp(
    const std::string& bwd_op_name) {
  auto& info_vector = OpMetaInfoMap::Instance()[name_];
  auto op_meta = OpMetaInfo(bwd_op_name);
  info_vector.emplace_back(std::move(op_meta));
  info_ptr_ = &(info_vector.back());
  return *this;
}

/////////////////////// Op register API /////////////////////////

void RegisterAllCustomOperator() {
  auto& op_meta_info_map = OpMetaInfoMap::Instance();
  framework::RegisterOperatorWithMetaInfoMap(op_meta_info_map);
}

void LoadCustomOperatorLib(const std::string& dso_name) {
  paddle::framework::LoadOpMetaInfoAndRegisterOp(dso_name);
}
}  // namespace paddle

extern "C" {

paddle::OpMetaInfoMap& PD_GetOpMetaInfoMap() {
  return paddle::OpMetaInfoMap::Instance();
}

}  // end extern "C"
