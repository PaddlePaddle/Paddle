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
#include <functional>
#include <memory>
#include <string>

namespace infrt {
namespace naive {

struct InferShapedKernelLauncher;

class InferShapedKernelRegistry {
 public:
  using InferShapeLauncherHandle = std::unique_ptr<InferShapedKernelLauncher>;
  using InferShapeLauncherCreator = std::function<InferShapeLauncherHandle()>;

  InferShapedKernelRegistry();

  void AddKernel(const std::string& key, InferShapeLauncherCreator&& creator);

  const InferShapeLauncherCreator& GetKernel(const std::string& key) const;

  size_t size() const;

  ~InferShapedKernelRegistry();

 private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
};

//! The global infershape registry.
InferShapedKernelRegistry* GetInferShapeRegistry();

}  // namespace naive
}  // namespace infrt

#define INFERSHAPED_KERNEL_CREATOR(infershape_launcher_class_)                 \
  []()                                                                         \
      -> ::infrt::naive::InferShapedKernelRegistry::InferShapeLauncherHandle { \
        return std::make_unique<infershape_launcher_class_>();                 \
      }
