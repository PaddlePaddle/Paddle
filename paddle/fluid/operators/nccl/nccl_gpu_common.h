/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace platform {
constexpr int kInvalidGPUId = -1;

struct Communicator {
  Communicator() {}

  int GetCommId(int device_id) const;

  void InitAll(const std::vector<int>& gpus);

  const std::vector<ncclComm_t>& comms() const;
};

}  // namespace platform
}  // namespace paddle
