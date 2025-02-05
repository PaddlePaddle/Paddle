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

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/platform/device_event_base.h"
#include "paddle/phi/core/tensor_utils.h"

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
    void CudaSynchronize();
    void CpuSynchronize();
    void UpdateWaitChain(const phi::DeviceContext& ctx);

   private:
    platform::DeviceEvent load_event_;  // event on offload stream
    Place task_place_;
  };

  std::shared_ptr<AsyncLoad::Task> Offload(phi::DenseTensor* dst,
                                           const phi::DenseTensor& src);

  std::shared_ptr<AsyncLoad::Task> OffloadWithOffset(
      phi::DenseTensor* dst,
      const phi::DenseTensor& src,
      size_t dst_offset,
      size_t src_offset,
      size_t offload_size);

  void PrepareLoadEnv(const std::string& key, const Place& place);
  void SyncCalcuStream(const Place& place,
                       phi::GPUContext* ctx,
                       platform::DeviceEvent& calc_event);  // NOLINT
  std::shared_ptr<AsyncLoad::Task> Reload(phi::DenseTensor* dst,
                                          const phi::DenseTensor& src);

 private:
  std::unordered_map<std::string, platform::DeviceEvent>
      place_to_calc_event_;  // event on calc stream
  bool is_initialized_{false};
  std::unique_ptr<phi::GPUContext> load_ctx_;
  Place gpu_place_;
  std::shared_ptr<AsyncLoad::Task> CreateTask(const Place& place);
};

}  //  namespace distributed
}  //  namespace paddle
