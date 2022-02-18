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

#include <algorithm>
#include "paddle/pten/core/enforce.h"
#include "paddle/pten/core/kernel_factory.h"

namespace paddle {
namespace framework {

void RegisterWithOpKernelInfoMap(
    const paddle::OpKernelInfoMap& op_kernel_info_map) {
  auto& kernel_info_map = op_kernel_info_map.GetMap();
  VLOG(3) << "[CUSTOM KERNEL] size of op_kernel_info_map: "
          << kernel_info_map.size();

  for (auto& pair : kernel_info_map) {
    VLOG(3) << "[CUSTOM KERNEL] try registering op name: " << pair.first;

    PADDLE_ENFORCE_EQ(
        pten::KernelFactory::Instance().HasCompatiblePtenKernel(pair.first),
        true,
        platform::errors::InvalidArgument(
            "[CUSTOM KERNEL] %s is not ready for custom kernel registering.",
            pair.first));

    for (auto& info_pair : pair.second) {
      VLOG(3) << "[CUSTOM KERNEL] try registering [" << pair.first << "]"
              << info_pair.first;

      auto& kernels = pten::KernelFactory::Instance().kernels();
      PADDLE_ENFORCE_EQ(
          kernels[pair.first].find(info_pair.first), kernels[pair.first].end(),
          platform::errors::InvalidArgument(
              "[CUSTOM KERNEL] The operator <%s>'s kernel: %s has been "
              "already existed in Paddle, please contribute PR if need "
              "to optimize the kernel code. Custom kernel do NOT support "
              "to replace existing kernel in Paddle.",
              pair.first, info_pair.first));

      kernels[pair.first][info_pair.first] = info_pair.second;

      VLOG(3) << "[CUSTOM KERNEL] Succeeded in registering operator <"
              << pair.first << ">'s kernel " << info_pair.first
              << " to Paddle. It will be used like native ones.";
    }
  }
}

void LoadCustomKernelLib(const std::string& dso_lib_path, void* dso_handle) {
#ifdef _LINUX
  typedef OpKernelInfoMap& get_op_kernel_info_map_t();
  auto* func = reinterpret_cast<get_op_kernel_info_map_t*>(
      dlsym(dso_handle, "PD_GetOpKernelInfoMap"));

  if (func == nullptr) {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path << "]: fail to find "
                 << "PD_GetOpKernelInfoMap symbol in this lib.";
    return;
  }
  auto& op_kernel_info_map = func();
  RegisterWithOpKernelInfoMap(op_kernel_info_map);
  LOG(INFO) << "Successed in loading custom kernels in lib: " << dso_lib_path;
#else
  VLOG(3) << "Unsupported: Custom kernel is only implemented on Linux.";
#endif
  return;
}

}  // namespace framework
}  // namespace paddle
