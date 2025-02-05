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

#if defined(PADDLE_WITH_XPU)
#include <memory>
#include <type_traits>
#include <vector>

#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#include "paddle/phi/core/platform/resource_pool.h"

namespace paddle {
namespace platform {

using XpuStreamObject = std::remove_pointer<xpuStream>::type;
using XpuEventObject = std::remove_pointer<xpuEventHandle>::type;

class XpuStreamResourcePool {
 public:
  std::shared_ptr<XpuStreamObject> New(int dev_idx);

  static XpuStreamResourcePool &Instance();

 private:
  XpuStreamResourcePool();

  DISABLE_COPY_AND_ASSIGN(XpuStreamResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<XpuStreamObject>>> pool_;
};

class XpuEventResourcePool {
 public:
  std::shared_ptr<XpuEventObject> New(int dev_idx);

  static XpuEventResourcePool &Instance();

 private:
  XpuEventResourcePool();

  DISABLE_COPY_AND_ASSIGN(XpuEventResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<XpuEventObject>>> pool_;
};

}  // namespace platform
}  // namespace paddle

#endif
