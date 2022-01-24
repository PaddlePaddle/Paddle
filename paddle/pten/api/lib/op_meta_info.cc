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

#include "paddle/pten/api/ext/op_meta_info.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/platform/enforce.h"

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
OpMetaInfo& OpMetaInfo::Attrs(std::vector<std::string>&& attrs) {
  attrs_ = std::forward<std::vector<std::string>>(attrs);
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

OpMetaInfoBuilder::OpMetaInfoBuilder(std::string&& name, size_t index) {
  // 1. member assign
  name_ = std::forward<std::string>(name);
  index_ = index;

  // 2. check and meta info build
  auto& info_vector = OpMetaInfoMap::Instance()[name_];
  // index check
  PADDLE_ENFORCE_EQ(
      info_vector.size(),
      index_,
      platform::errors::PreconditionNotMet(
          "The operator %s's meta info register failed. "
          "Please make sure you call marcos as order `PD_BUILD_OP`, "
          "`PD_BUILD_GRAD_OP`, `PD_BUILD_DOUBLE_GRAD_OP`.",
          name_));
  switch (index_) {
    case 0:
      break;
    case 1:
      name_ = name_ + "_grad";
      break;
    case 2:
      name_ = name_ + "_grad_grad";
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Not support index `%d` when construct OpMetaInfoBuilder, "
          "now only support `0, 1, 2`.",
          index_));
  }
  auto op_meta = OpMetaInfo(name_);
  info_vector.emplace_back(std::move(op_meta));
  // 3. get current info ptr
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

OpMetaInfoBuilder& OpMetaInfoBuilder::Attrs(std::vector<std::string>&& attrs) {
  info_ptr_->Attrs(std::forward<std::vector<std::string>>(attrs));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetKernelFn(KernelFunc func) {
  info_ptr_->SetKernelFn(std::forward<KernelFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInferShapeFn(InferShapeFunc func) {
  info_ptr_->SetInferShapeFn(std::forward<InferShapeFunc>(func));
  return *this;
}

OpMetaInfoBuilder& OpMetaInfoBuilder::SetInferDtypeFn(InferDtypeFunc func) {
  PADDLE_ENFORCE_EQ(
      index_,
      0UL,
      platform::errors::Unimplemented(
          "Currently, the InferDtypeFn setting of Grad Op is not supported, "
          "And backward Tensor `X@GRAD` will use the dtype of forward Tensor "
          "`X` by default."));
  info_ptr_->SetInferDtypeFn(std::forward<InferDtypeFunc>(func));
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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _WIN32
// C-API to get global OpMetaInfoMap.
paddle::OpMetaInfoMap& PD_GetOpMetaInfoMap() {
  return paddle::OpMetaInfoMap::Instance();
}
#endif

#ifdef __cplusplus
}  // end extern "C"
#endif
