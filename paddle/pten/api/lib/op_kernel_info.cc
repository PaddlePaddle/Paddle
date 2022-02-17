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

//////////////// Op Kernel Info Map /////////////////

const pten::KernelNameMap& OpKernelInfoMap::GetMap() const { return kernels_; }

/////////////////////// Op register API /////////////////////////

// For inference: compile directly with framework
// Call after PD_REGISTER_KERNEL(...)
void RegisterAllCustomKernel() {
  auto& op_kernel_info_map = OpKernelInfoMap::Instance();
  framework::RegisterWithOpKernelInfoMap(op_kernel_info_map);
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
