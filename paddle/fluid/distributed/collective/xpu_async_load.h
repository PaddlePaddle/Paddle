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
// limitations under the License.#pragma once

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

// #include "paddle/fluid/platform/device_context.h"
// #include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/platform/device_event_base.h"
// #include "paddle/phi/core/places.h"

namespace paddle {
namespace distributed {

using Place = phi::Place;

/**
 * AsyncLoad that does NOT use platform::DeviceEvent if place == XPU.
 */
class XpuAsyncLoad {
 public:
  class Task {
   public:
    explicit Task(const Place& place);
    virtual ~Task();

    bool IsCompleted();

    // Replaces CudaSynchronize with XpuSynchronize
    void XpuSynchronize();
    void CpuSynchronize();

    // If not XPU, record the event. If XPU, do nothing
    void UpdateWaitChain(const phi::DeviceContext& ctx);

   private:
    bool use_event_;  // false if place is XPU
    platform::DeviceEvent load_event_;
    Place task_place_;
  };

  // Offload
  std::shared_ptr<Task> Offload(phi::DenseTensor* dst,
                                const phi::DenseTensor& src);

  // OffloadWithOffset
  std::shared_ptr<Task> OffloadWithOffset(phi::DenseTensor* dst,
                                          const phi::DenseTensor& src,
                                          size_t dst_offset,
                                          size_t src_offset,
                                          size_t offload_size);

  // Reload
  std::shared_ptr<Task> Reload(phi::DenseTensor* dst,
                               const phi::DenseTensor& src);

 private:
  bool is_initialized_{false};

  // A fallback "offload context," though we won't do multi-stream sync for XPU
  std::unique_ptr<phi::XPUContext> load_ctx_;
  Place xpu_place_;

  std::shared_ptr<Task> CreateTask(const Place& place);

  // If not XPU, store calc-event. If XPU, skip
  std::unordered_map<std::string, platform::DeviceEvent> place_to_calc_event_;

  // Prepare env
  void PrepareLoadEnv(const std::string& key, const Place& place);

  // If not XPU, do event sync. If XPU, skip
  void SyncCalcuStream(const Place& place,
                       phi::XPUContext* offload_ctx,
                       platform::DeviceEvent* calc_event);
};

}  // namespace distributed
}  // namespace paddle
