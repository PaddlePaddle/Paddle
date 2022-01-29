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

#include "paddle/infrt/host_context/kernel_registry.h"

#include <unordered_map>

#include "glog/logging.h"
#include "llvm/ADT/SmallVector.h"

namespace infrt {
namespace host_context {

struct KernelRegistry::Impl {
  std::unordered_map<std::string, KernelImplementation> data;
  std::unordered_map<std::string, llvm::SmallVector<std::string, 4>> attr_names;
};

KernelRegistry::KernelRegistry() : impl_(std::make_unique<Impl>()) {}

void KernelRegistry::AddKernel(const std::string &key,
                               KernelImplementation fn) {
  CHECK(!impl_->data.count(key)) << "kernel [" << key
                                 << "] is registered twice";
  impl_->data.emplace(key, fn);
}

void KernelRegistry::AddKernelAttrNameList(
    const std::string &key, const std::vector<std::string> &names) {
  CHECK(!impl_->attr_names.count(key))
      << "kernel [" << key << "] is registered twice in attribute names";
  impl_->attr_names.emplace(
      key, llvm::SmallVector<std::string, 4>(names.begin(), names.end()));
}

KernelImplementation KernelRegistry::GetKernel(const std::string &key) const {
  auto it = impl_->data.find(key);
  return it != impl_->data.end() ? it->second : KernelImplementation{};
}

std::vector<std::string> KernelRegistry::GetKernelList() const {
  std::vector<std::string> res(impl_->data.size());
  for (auto i : impl_->data) {
    res.push_back(i.first);
  }
  return res;
}

KernelRegistry::~KernelRegistry() {}

size_t KernelRegistry::size() const { return impl_->data.size(); }

KernelRegistry *GetCpuKernelRegistry() {
  static auto registry = std::make_unique<KernelRegistry>();
  return registry.get();
}

}  // namespace host_context
}  // namespace infrt
