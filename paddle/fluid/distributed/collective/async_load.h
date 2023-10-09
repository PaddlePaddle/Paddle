// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/eager/api/utils/tensor_utils.h"  // NOTE: this header is required somewhere
#include "paddle/fluid/platform/device_event_base.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {

using Place = phi::Place;

class AsyncLoad {
 public:
  class Task {
   public:
    explicit Task(const Place& place);
    virtual ~Task();
    bool IsCompleted();
    void Synchronize();
    void UpdateWaitChain(const phi::DeviceContext& ctx);

   private:
    platform::DeviceEvent load_event_;  // event on offload stream
    Place task_place_;
  };

  // AsyncLoad();

  std::shared_ptr<AsyncLoad::Task> Offload(phi::DenseTensor* dst,
                                           const phi::DenseTensor& src);

  // void SyncCalcStream();

 private:
  // platform::DeviceEvent calc_event_;
  std::unordered_map<std::string, platform::DeviceEvent>
      place_to_calc_event_;  // event on calc stream

  std::unordered_map<std::string, std::unique_ptr<phi::GPUContext>>
      place_to_load_ctx_;

  std::shared_ptr<AsyncLoad::Task> CreateTask(const Place& place);
};

}  //  namespace distributed
}  //  namespace paddle
