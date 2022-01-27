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

#include "paddle/infrt/naive/infershape/infershape_registry.h"

#include <unordered_map>

#include "paddle/infrt/naive/infershape/infershape_launcher.h"

namespace infrt {
namespace naive {

struct InferShapeRegistry::Impl {
  std::unordered_map<std::string, InferShapeLauncherCreator> data;
};

void InferShapeRegistry::AddKernel(
    const std::string& key,
    InferShapeRegistry::InferShapeLauncherCreator&& creator) {
  CHECK(!impl_->data.count(key)) << "Item called " << key << " duplicates";
  impl_->data.emplace(key, std::move(creator));
}

const InferShapeRegistry::InferShapeLauncherCreator&
InferShapeRegistry::GetKernel(const std::string& key) const {
  auto it = impl_->data.find(key);
  CHECK(it != impl_->data.end()) << "No item called " << key << " exists";
  return it->second;
}

size_t InferShapeRegistry::size() const { return impl_->data.size(); }

InferShapeRegistry* GetInferShapeRegistry() {
  static auto registry = std::make_unique<InferShapeRegistry>();
  return registry.get();
}

}  // namespace naive
}  // namespace infrt
