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

#include "paddle/infrt/tensor/phi/tensor_map.h"

#include "glog/logging.h"
#include "llvm/Support/ErrorHandling.h"

namespace infrt {
namespace phi {

void DenseTensorMap::SetDenseTensor(const std::string& name,
                                    std::unique_ptr<::Tensor>&& tensor) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = map_.emplace(std::make_pair(name, std::move(tensor)));
  if (!it.second) {
    llvm_unreachable("dense tensor map insert failed.");
  }
}

::Tensor* DenseTensorMap::GetDenseTensor(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = map_.find(name);
  if (it != map_.end()) {
    return it->second.get();
  }
  LOG(WARNING) << "can not find `" << name << "` in the tensor map.";
  return nullptr;
}

size_t DenseTensorMap::size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return map_.size();
}

}  // namespace phi
}  // namespace infrt
