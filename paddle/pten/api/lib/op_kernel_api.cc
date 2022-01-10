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

#include "paddle/pten/api/ext/op_kernel_api.h"
#include "paddle/fluid/framework/custom_kernel.h"  // todo
#include "paddle/fluid/framework/operator.h"

namespace paddle {

////////////////////// Op Kernel Info //////////////////////

OpKernelInfo& OpKernelInfo::SetKernelFn(KernelFunc&& func) {
  kernel_fn_ = std::forward<KernelFunc>(func);
  return *this;
}

OpKernelInfo& OpKernelInfo::SetPtenKernelFn(PtenKernelFunc&& func) {
  pten_kernel_fn_ = std::forward<PtenKernelFunc>(func);
  return *this;
}

OpKernelInfo& OpKernelInfo::SetPtenVariadicFn(void* func) {
  variadic_fn_ = func;
  return *this;
}

OpKernelInfo& OpKernelInfo::SetPtenArgsParseFn(PtenKernelArgsParseFn&& func) {
  pten_kernel_args_parse_fn_ = func;
  return *this;
}

OpKernelInfo& OpKernelInfo::SetPtenArgsDefFn(PtenKernelArgsDefFn&& func) {
  pten_kernel_args_def_fn_ = func;
  return *this;
}
//////////////// Op Kernel Info Map /////////////////

std::vector<OpKernelInfo>& OpKernelInfoMap::operator[](
    const std::string& name) {
  return map_[name];
}

const std::unordered_map<std::string, std::vector<OpKernelInfo>>&
OpKernelInfoMap::GetMap() const {
  return map_;
}

//////////////// Op Kernel Info Builder /////////////////

OpKernelInfoBuilder::OpKernelInfoBuilder(std::string&& op_name,
                                         pten::Backend backend,
                                         pten::DataLayout data_layout,
                                         pten::DataType data_type) {
  // 1. member assign
  op_name_ = std::forward<std::string>(op_name);
  backend_ = backend;
  layout_ = data_layout;
  dtype_ = data_type;

  // 2. check and meta info build
  auto& info_vector = OpKernelInfoMap::Instance()[op_name_];
  auto kernel_meta = OpKernelInfo(op_name_, backend_, layout_, dtype_);
  info_vector.emplace_back(std::move(kernel_meta));

  // 3. get current info ptr
  info_ptr_ = &(info_vector.back());
}

OpKernelInfoBuilder& OpKernelInfoBuilder::SetKernelFn(KernelFunc func) {
  info_ptr_->SetKernelFn(std::forward<KernelFunc>(func));
  return *this;
}

OpKernelInfoBuilder& OpKernelInfoBuilder::SetPtenKernelFn(PtenKernelFunc func) {
  info_ptr_->SetPtenKernelFn(std::forward<PtenKernelFunc>(func));
  return *this;
}

OpKernelInfoBuilder& OpKernelInfoBuilder::SetPtenVariadicFn(void* func) {
  info_ptr_->SetPtenVariadicFn(func);
  return *this;
}

OpKernelInfoBuilder& OpKernelInfoBuilder::SetPtenArgsParseFn(
    PtenKernelArgsParseFn func) {
  info_ptr_->SetPtenArgsParseFn(std::forward<PtenKernelArgsParseFn>(func));
  return *this;
}

OpKernelInfoBuilder& OpKernelInfoBuilder::SetPtenArgsDefFn(
    PtenKernelArgsDefFn func) {
  info_ptr_->SetPtenArgsDefFn(std::forward<PtenKernelArgsDefFn>(func));
  return *this;
}
/////////////////////// Op register API /////////////////////////

}  // namespace paddle

// C-API to get global OpKernelInfoMap.
paddle::OpKernelInfoMap& PD_GetOpKernelInfoMap() {
  return paddle::OpKernelInfoMap::Instance();
}
