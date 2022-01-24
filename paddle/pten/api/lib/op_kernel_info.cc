/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/api/ext/op_kernel_info.h"
#include "paddle/fluid/framework/custom_kernel.h"

namespace paddle {

////////////////////// Op Kernel Info //////////////////////

OpKernelInfo& OpKernelInfo::SetKernelFn(CustomKernelFunc&& func) {
  kernel_fn_ = std::forward<CustomKernelFunc>(func);
  return *this;
}

OpKernelInfo& OpKernelInfo::SetVariadicKernelFn(void* func) {
  variadic_kernel_fn_ = func;
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

  // 2. info parse
  auto& info_vector = OpKernelInfoMap::Instance()[op_name_];
  auto op_kernel_info = OpKernelInfo(op_name_, backend_, layout_, dtype_);
  info_vector.emplace_back(std::move(op_kernel_info));

  // 3. get current info ptr
  info_ptr_ = &(info_vector.back());
}

OpKernelInfoBuilder& OpKernelInfoBuilder::SetKernelFn(CustomKernelFunc func) {
  info_ptr_->SetKernelFn(std::forward<CustomKernelFunc>(func));
  return *this;
}

OpKernelInfoBuilder& OpKernelInfoBuilder::SetVariadicKernelFn(void* func) {
  info_ptr_->SetVariadicKernelFn(func);
  return *this;
}

OpKernelInfoBuilder& OpKernelInfoBuilder::ArgsParse(
    CustomKernelArgsParseFn func) {
  func(this->info_ptr_);
  return *this;
}

OpKernelInfoBuilder& OpKernelInfoBuilder::ArgsDef(CustomKernelArgsDefFn func) {
  func(this->info_ptr_);
  return *this;
}

/////////////////////// Op register API /////////////////////////

// For inference: compile directly with framework
// Call after PD_REGISTER_KERNEL(...)
void RegisterAllCustomKernel() {
  auto& op_kernel_info_map = OpKernelInfoMap::Instance();
  framework::RegisterKernelWithMetaInfoMap(op_kernel_info_map);
}

// Using this api to load compiled custom kernel's dynamic library and
// register custom kernels
void LoadCustomKernelLib(const std::string& dso_name) {
  framework::LoadCustomKernelLib(dso_name);
}

}  // namespace paddle

#ifdef __cplusplus
extern "C" {
#endif

// C-API to get global OpKernelInfoMap.
paddle::OpKernelInfoMap& PD_GetOpKernelInfoMap() {
  return paddle::OpKernelInfoMap::Instance();
}

#ifdef __cplusplus
}  // end extern "C"
#endif
