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
#include <dirent.h>
#include <algorithm>
#include <regex>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/pten/api/ext/op_kernel_info.h"
#include "paddle/pten/core/compat/convert_utils.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_registry.h"

namespace paddle {

namespace framework {

void RegisterKernelWithMetaInfoMap(
    const paddle::OpKernelInfoMap& op_kernel_info_map) {
  auto& kernel_info_map = op_kernel_info_map.GetMap();
  VLOG(3) << "[CUSTOM KERNEL] size of op_kernel_info_map: "
          << kernel_info_map.size();

  for (auto& pair : kernel_info_map) {
    VLOG(3) << "[CUSTOM KERNEL] try registering op name: " << pair.first;

    // 1.Check whether this kernel is valid in pten
    PADDLE_ENFORCE_EQ(
        pten::KernelFactory::Instance().HasCompatiblePtenKernel(pair.first),
        true,
        platform::errors::InvalidArgument(
            "[CUSTOM KERNEL] %s is not ready for custom kernel registering.",
            pair.first));

    for (auto& info_pair : pair.second) {
      VLOG(3) << "[CUSTOM KERNEL] try registering [" << pair.first << "]"
              << info_pair.first;

      // 2.Check whether kernel_key has been already registed
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

void LoadCustomKernelLib(const std::string& dso_lib_path) {
#ifdef _LINUX
  void* dso_handle = nullptr;
  int dynload_flags = RTLD_NOW | RTLD_LOCAL;
  dso_handle = dlopen(dso_lib_path.c_str(), dynload_flags);

  // MUST valid dso_lib_path
  PADDLE_ENFORCE_NOT_NULL(
      dso_handle,
      platform::errors::InvalidArgument(
          "Fail to open library: %s with error: %s", dso_lib_path, dlerror()));

  typedef OpKernelInfoMap& get_op_kernel_info_map_t();
  auto* func = reinterpret_cast<get_op_kernel_info_map_t*>(
      dlsym(dso_handle, "PD_GetOpKernelInfoMap"));

  if (func == nullptr) {
    LOG(INFO) << "Skipped lib [" << dso_lib_path << "]: fail to find "
              << "PD_GetOpKernelInfoMap symbol in this lib.";
    return;
  }
  auto& op_kernel_info_map = func();
  RegisterKernelWithMetaInfoMap(op_kernel_info_map);
  LOG(INFO) << "Successed in loading custom kernels in lib: " << dso_lib_path;
#else
  VLOG(3) << "Unsupported: Custom kernel is only implemented on Linux.";
#endif
  return;
}

// List all libs with given path
std::vector<std::string> ListAllLib(const std::string& libs_path) {
  DIR* dir = nullptr;
  dir = opendir(libs_path.c_str());

  // MUST valid libs_path
  PADDLE_ENFORCE_NOT_NULL(dir, platform::errors::InvalidArgument(
                                   "Fail to open path: %s", libs_path));

  dirent* ptr = nullptr;
  std::vector<std::string> libs;
  std::regex express(".*\\.so");
  std::match_results<std::string::iterator> results;
  while ((ptr = readdir(dir)) != nullptr) {
    std::string filename(ptr->d_name);
    if (std::regex_match(filename.begin(), filename.end(), results, express)) {
      libs.emplace_back(libs_path + '/' + filename);
      LOG(INFO) << "Found lib [" << filename << "]";
    } else {
      VLOG(3) << "Skipped file [" << filename << "] without .so postfix";
    }
  }
  closedir(dir);
  return libs;
}

// Load custom kernels with given path
void LoadCustomKernel(const std::string& libs_path) {
  VLOG(3) << "Try loading custom libs from: [" << libs_path << "]";
  std::vector<std::string> libs = ListAllLib(libs_path);
  for (auto& lib_path : libs) {
    LoadCustomKernelLib(lib_path);
  }
  LOG(INFO) << "Finished in LoadCustomKernel with libs_path: [" << libs_path
            << "]";
}

}  // namespace framework
}  // namespace paddle
