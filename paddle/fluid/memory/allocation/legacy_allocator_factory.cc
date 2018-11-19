// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/legacy_allocator.h"
#include "paddle/fluid/memory/allocation/allocator_factory.h"
#include "paddle/fluid/memory/allocation/allocator_factory_registry.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/platform/gpu_info.h"
namespace paddle {
namespace memory {
namespace allocation {

class LegacyAllocatorFactory : public AllocatorFactory {
 public:
  std::map<platform::Place, std::unique_ptr<Allocator>> Build() const override {
    std::vector<platform::Place> places{platform::CPUPlace()};
#ifdef PADDLE_WITH_CUDA
    for (int dev_id = 0; dev_id < platform::GetCUDADeviceCount(); ++dev_id) {
      places.emplace_back(platform::CUDAPlace(dev_id));
    }
    places.emplace_back(platform::CUDAPinnedPlace());
#endif
    std::map<platform::Place, std::unique_ptr<Allocator>> allocators;
    for (auto& p : places) {
      allocators.emplace(p, std::unique_ptr<Allocator>(new LegacyAllocator(p)));
    }

    return allocators;
  }

  bool CanBuild() const override {
    return AllocatorStrategy::kLegacy == GetAllocatorStrategy();
  }
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle

REGISTER_ALLOCATOR_FACTORY(legacy,
                           paddle::memory::allocation::LegacyAllocatorFactory);
