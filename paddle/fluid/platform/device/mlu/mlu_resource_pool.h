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

#if defined(PADDLE_WITH_MLU)
#include <memory>
#include <type_traits>
#include <vector>

#include "paddle/fluid/platform/device/mlu/mlu_info.h"
#include "paddle/fluid/platform/resource_pool.h"

namespace paddle {
namespace platform {

using MluStreamObject = std::remove_pointer<mluStream>::type;
using MluEventObject = std::remove_pointer<mluEventHandle>::type;

class MluStreamResourcePool {
 public:
  std::shared_ptr<MluStreamObject> New(int dev_idx);

  static MluStreamResourcePool &Instance();

 private:
  MluStreamResourcePool();

  DISABLE_COPY_AND_ASSIGN(MluStreamResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<MluStreamObject>>> pool_;
};

class MluEventResourcePool {
 public:
  std::shared_ptr<MluEventObject> New(int dev_idx);

  static MluEventResourcePool &Instance();

 private:
  MluEventResourcePool();

  DISABLE_COPY_AND_ASSIGN(MluEventResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<MluEventObject>>> pool_;
};

}  // namespace platform
}  // namespace paddle

#endif
