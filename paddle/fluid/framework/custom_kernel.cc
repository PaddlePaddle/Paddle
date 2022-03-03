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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include "paddle/fluid/framework/custom_kernel.h"
#include "paddle/phi/core/custom_kernel.h"

namespace paddle {
namespace framework {

void LoadCustomKernelLib(const std::string& dso_lib_path, void* dso_handle) {
#ifdef _LINUX
  typedef phi::CustomKernelMap& get_custom_kernel_map_t();
  auto* func = reinterpret_cast<get_custom_kernel_map_t*>(
      dlsym(dso_handle, "PD_GetCustomKernelMap"));

  if (func == nullptr) {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path << "]: fail to find "
                 << "PD_GetCustomKernelMap symbol in this lib.";
    return;
  }
  auto& custom_kernel_map = func();
  phi::RegisterCustomKernels(custom_kernel_map);
  LOG(INFO) << "Successed in loading custom kernels in lib: " << dso_lib_path;
#else
  VLOG(3) << "Unsupported: Custom kernel is only implemented on Linux.";
#endif
  return;
}

}  // namespace framework
}  // namespace paddle
