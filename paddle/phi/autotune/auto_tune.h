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

#include <unordered_map>
#include "glog/logging.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_factory.h"

namespace phi {

/* This class is the main control abstraction of op auto-tune, to judge whether
   the op needed auto-tune, and choose out the best performance kernel
   implement.
   The main function of this class is below :
        1. Judge whether the op need auto-tune.
        2. Tunning the op with different kernels
        3. Choose out and cache the best kernel implement.
*/
class AutoTunerBase {
 public:
  template <typename Context>
  phi::Kernel PickBestAlgorithm(Context ctx,
                                const std::string& op_name,
                                bool need_workspace = false) {
    float min_time = -1.f;
    size_t aviliable_memory = 0;
    if (need_workspace) {
      // Query the avaliable memory info
      aviliable_memory = paddle::platform::GpuAvailableMemToAlloc();
    }
    if (kernels_.empty()) {
      return phi::Kernel();
    }

    for (&kernel : kernels) {
      if (need_workspace) {
        /* TODO(limingshu): Currently, only conv need workspace and workspace
           query has beed added into conv, if more op need workspace, the
           function will be achieved. */
      }
      float kernel_time = RunKernelSync(kernel);
      if (min_time > 0 && kernel_time < min_time) {
        min_time = kernel_time;
        Kernel selected_kernel = kernel;
      }
    }
  }

  void KernelCollection(const std::string& op_name) {
    auto has_kernel =
        std::find(op_list_.begin(), op_list_.end(), op_name) != op_list_.end();
    if (has_kernel) {
      if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(op_name)) {
        KernelKeyMap kernel_iter =
            phi::KernelFactory::Instance().SelectKernelMap(op_name);
        PADDLE_ENFORCE_NE(kernel_iter,
                          phi::KernelKeyMap(),
                          platform::errors::NotFound(
                              "Cannot find %s op gpu kernels.", op_name));

        for (auto iter = kernel_iter.begin(); iter != kernel_iter.end();
             iter++) {
          auto kernel_key = iter->first;
          if (kernel_key.backend() == phi::Backend::GPU ||
              kernel_key.backend() == phi::Backend::GPUDNN) {
            kernels_.emplace_back(iter->second);
          }
        }
      }
    }
  }

  template <typename Context, typename Callback>
  float RunKernelSync(Context ctx, Callback&& kernel_func) {
    /*
    */
  }

 private:
  std::vector<phi::Kernel> kernels_;
  std::vector<string> op_list_{"conv2d"};
};

}  // namespace phi
