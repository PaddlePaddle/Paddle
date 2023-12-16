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

#pragma once

#include "paddle/common/macros.h"
#include "paddle/phi/core/kernel_factory.h"

namespace phi {
/**
 * Note:
 * Used to store kernels' info before registered to KernelFactory.
 */
class CustomKernelMap {
 public:
  static CustomKernelMap& Instance() {
    static CustomKernelMap g_custom_kernel_info_map;
    return g_custom_kernel_info_map;
  }

  void RegisterCustomKernel(const std::string& kernel_name,
                            const KernelKey& kernel_key,
                            const Kernel& kernel);

  void RegisterCustomKernels();

  KernelNameMap& Kernels() { return kernels_; }

  const KernelNameMap& GetMap() const { return kernels_; }

 private:
  CustomKernelMap() = default;
  DISABLE_COPY_AND_ASSIGN(CustomKernelMap);

  KernelNameMap kernels_;
};

}  // namespace phi
