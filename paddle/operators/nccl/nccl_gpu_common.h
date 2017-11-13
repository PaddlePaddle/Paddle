/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/platform/device_context.h"
#include "paddle/platform/dynload/nccl.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace platform {

constexpr int kInvalidGPUId = -1;

struct Communicator {
  std::vector<ncclComm_t> comms_;
  std::unordered_map<int, int> comm_id_map_;
  bool inited_;

  Communicator() {}

  int GetCommId(int device_id) const { return comm_id_map_.at(device_id); }

  void InitAll(const std::vector<int>& gpus) {
    comms_.resize(gpus.size());
    inited_ = false;
    for (size_t i = 0; i < gpus.size(); ++i) {
      comm_id_map_[gpus[i]] = i;
    }
    PADDLE_ENFORCE(
        dynload::ncclCommInitAll(comms_.data(), gpus.size(), gpus.data()));
    inited_ = true;
  }

  ~Communicator() {
    if (inited_) {
      for (size_t i = 0; i < comms_.size(); ++i) {
        // FIXME(dzh) : PADDLE_ENFORCE return void
        dynload::ncclCommDestroy(comms_[i]);
      }
    }
  }

  DISABLE_COPY_AND_ASSIGN(Communicator);
};

}  // namespace platform
}  // namespace paddle
