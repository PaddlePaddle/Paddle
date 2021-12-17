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

OpKernelInfoBuilder::OpKernelInfoBuilder(std::string&& name) {
  // 1. member assign
  name_ = std::forward<std::string>(name);
  // 2. check and meta info build
  auto& info_vector = OpKernelInfoMap::Instance()[name_];

  auto op_meta = OpKernelInfo(name_);
  info_vector.emplace_back(std::move(op_meta));
  // 3. get current info ptr
  info_ptr_ = &(info_vector.back());
}

OpKernelInfoBuilder& OpKernelInfoBuilder::SetKernelFn(KernelFunc func) {
  info_ptr_->SetKernelFn(std::forward<KernelFunc>(func));
  return *this;
}

/////////////////////// Op register API /////////////////////////

}  // namespace paddle

// C-API to get global OpKernelInfoMap.
paddle::OpKernelInfoMap& PD_GetOpKernelInfoMap() {
  return paddle::OpKernelInfoMap::Instance();
}

// int PD_NumInputs(const paddle::PD_ExecutionContext* ctx) {
//   auto* cc_ctx = reinterpret_cast<paddle::framework::ExecutionContext*>(
//       const_cast<paddle::PD_ExecutionContext*>(ctx));
//   auto innamelist = cc_ctx->InNameList();
//   for (auto& input : innamelist) {
//     std::cout << "PD_NumInputs: " << input << std::endl;
//   }
//   return static_cast<int>(innamelist.size());
// }
