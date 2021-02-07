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

#include "paddle/fluid/framework/op_meta_info_helper.h"

namespace paddle {

////////////////////// Op Meta Info //////////////////////

OpMetaInfo& OpMetaInfo::Inputs(std::vector<std::string>&& inputs) {
  inputs_ = inputs;
  return *this;
}
OpMetaInfo& OpMetaInfo::Outputs(std::vector<std::string>&& outputs) {
  outputs_ = outputs;
  return *this;
}
OpMetaInfo& OpMetaInfo::SetKernelFn(KernelFunc&& func) {
  kernel_fn_ = func;
  return *this;
}
OpMetaInfo& OpMetaInfo::SetInferShapeFn(InferShapeFunc&& func) {
  infer_shape_fn_ = func;
  return *this;
}
OpMetaInfo& OpMetaInfo::SetInferDtypeFn(InferDtypeFunc&& func) {
  infer_dtype_fn_ = func;
  return *this;
}

////////////////////// Op Meta Info Helper //////////////////////

const std::string& OpMetaInfoHelper::GetOpName(const OpMetaInfo& info) {
  return info.name_;
}
const std::vector<std::string>& OpMetaInfoHelper::GetInputs(
    const OpMetaInfo& info) {
  return info.inputs_;
}
const std::vector<std::string>& OpMetaInfoHelper::GetOutputs(
    const OpMetaInfo& info) {
  return info.outputs_;
}
const std::vector<std::string>& OpMetaInfoHelper::GetAttrs(
    const OpMetaInfo& info) {
  return info.attrs_;
}
const KernelFunc& OpMetaInfoHelper::GetKernelFn(const OpMetaInfo& info) {
  return info.kernel_fn_;
}
const InferShapeFunc& OpMetaInfoHelper::GetInferShapeFn(
    const OpMetaInfo& info) {
  return info.infer_shape_fn_;
}
const InferDtypeFunc& OpMetaInfoHelper::GetInferDtypeFn(
    const OpMetaInfo& info) {
  return info.infer_dtype_fn_;
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
  info_vector.emplace_back(op_meta);
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
  info_vector.emplace_back(op_meta);
  info_ptr_ = &(info_vector.back());
  return *this;
}

}  // namespace paddle
