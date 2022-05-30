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

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "paddle/fluid/platform/resource_pool.h"

namespace paddle {
namespace platform {

using NpuStreamObject = std::remove_pointer<aclrtStream>::type;
using NpuEventObject = std::remove_pointer<aclrtEvent>::type;

class NpuStreamResourcePool {
 public:
  std::shared_ptr<NpuStreamObject> New(int dev_idx);

  static NpuStreamResourcePool &Instance();

 private:
  NpuStreamResourcePool();

  DISABLE_COPY_AND_ASSIGN(NpuStreamResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<NpuStreamObject>>> pool_;
};

class NpuEventResourcePool {
 public:
  std::shared_ptr<NpuEventObject> New(int dev_idx);

  static NpuEventResourcePool &Instance();

 private:
  NpuEventResourcePool();

  DISABLE_COPY_AND_ASSIGN(NpuEventResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<NpuEventObject>>> pool_;
};

}  // namespace platform
}  // namespace paddle

#endif
