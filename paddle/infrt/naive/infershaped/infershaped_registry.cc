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

#include "paddle/infrt/naive/infershaped/infershaped_registry.h"

#include <unordered_map>

#include "paddle/infrt/naive/infershaped/infershaped_kernel_launcher.h"

namespace infrt {
namespace naive {

struct InferShapedKernelRegistry::Impl {
  std::unordered_map<std::string, InferShapeLauncherCreator> data;
};

InferShapedKernelRegistry::InferShapedKernelRegistry()
    : impl_(std::make_unique<Impl>()) {}

void InferShapedKernelRegistry::AddKernel(
    const std::string& key,
    InferShapedKernelRegistry::InferShapeLauncherCreator&& creator) {
  CHECK(!impl_->data.count(key)) << "Item called " << key << " duplicates";
  impl_->data.emplace(key, std::move(creator));
}

const InferShapedKernelRegistry::InferShapeLauncherCreator&
InferShapedKernelRegistry::GetKernel(const std::string& key) const {
  auto it = impl_->data.find(key);
  CHECK(it != impl_->data.end()) << "No item called " << key << " exists";
  return it->second;
}

size_t InferShapedKernelRegistry::size() const { return impl_->data.size(); }

InferShapedKernelRegistry* GetInferShapeRegistry() {
  static auto registry = std::make_unique<InferShapedKernelRegistry>();
  return registry.get();
}

InferShapedKernelRegistry::~InferShapedKernelRegistry() {}

}  // namespace naive
}  // namespace infrt
