// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>

namespace infrt {
namespace host_context {

class KernelFrame;

using KernelImplementation = void (*)(KernelFrame *frame);

/**
 * Hold the kernels registered in the system.
 */
class KernelRegistry {
 public:
  KernelRegistry();

  void AddKernel(const std::string &key, KernelImplementation fn);
  void AddKernelAttrNameList(const std::string &key,
                             const std::vector<std::string> &names);

  KernelImplementation GetKernel(const std::string &key) const;
  std::vector<std::string> GetKernelList() const;

  size_t size() const;

  ~KernelRegistry();

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

//! The global CPU kernel registry.
KernelRegistry *GetCpuKernelRegistry();

}  // namespace host_context
}  // namespace infrt

/**
 * compile function RegisterKernels in C way to avoid C++ name mangling.
 */
#ifdef __cplusplus
extern "C" {
#endif
void RegisterKernels(infrt::host_context::KernelRegistry *registry);
#ifdef __cplusplus
}
#endif
