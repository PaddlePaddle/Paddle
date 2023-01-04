// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/custom_kernel.h"

#include "glog/logging.h"

static std::vector<std::string> gpu_exclusive_kernels({"sync_batch_norm",
                                                       "sync_batch_norm_grad"});

namespace phi {

void CustomKernelMap::RegisterCustomKernel(const std::string& name,
                                           const KernelKey& key,
                                           const Kernel& kernel) {
  PADDLE_ENFORCE_EQ(kernels_[name].find(key),
                    kernels_[name].end(),
                    phi::errors::AlreadyExists(
                        "The custom kernel [%s:%s] has been already existed in "
                        "CustomKernelMap, please check if any duplicate kernel "
                        "info in your lib(s) before load again.",
                        name,
                        key));
  kernels_[name][key] = kernel;
}

void CustomKernelMap::RegisterCustomKernels() {
  VLOG(3) << "Size of custom_kernel_map: " << kernels_.size();

  if (kernels_.size() <= 0) {
    LOG(INFO) << "No custom kernel info found in loaded lib(s).";
    return;
  }
  auto& kernels = KernelFactory::Instance().kernels();
  for (auto& pair : kernels_) {
    if (kernels.find(pair.first) == kernels.cend()) {
      if (std::find(gpu_exclusive_kernels.cbegin(),
                    gpu_exclusive_kernels.cend(),
                    pair.first) == gpu_exclusive_kernels.cend()) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "The kernel %s is not ready for custom kernel registering.",
            pair.first));
      }
    }

    for (auto& info_pair : pair.second) {
      PADDLE_ENFORCE_EQ(
          kernels[pair.first].find(info_pair.first),
          kernels[pair.first].end(),
          phi::errors::AlreadyExists(
              "The kernel [%s:%s] has been already existed "
              "in Paddle, please contribute PR if it is necessary "
              "to optimize the kernel code. Custom kernel does NOT support "
              "to replace existing kernel in Paddle.",
              pair.first,
              info_pair.first));

      kernels[pair.first][info_pair.first] = info_pair.second;

      VLOG(3) << "Successed in registering kernel [" << pair.first << ":"
              << info_pair.first
              << "] to Paddle. It will be used like native ones.";
    }
  }
  LOG(INFO) << "Successed in loading " << kernels_.size()
            << " custom kernel(s) from loaded lib(s), will be "
            << "used like native ones.";
  kernels_.clear();
}

}  // namespace phi
