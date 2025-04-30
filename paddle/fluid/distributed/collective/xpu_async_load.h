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

#include "paddle/common/errors.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/fluid/distributed/collective/bkcl_tools.h"
#include "paddle/fluid/distributed/collective/common.h"

// Include the allocator facade header.
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

namespace paddle {
namespace distributed {

using Place = phi::Place;

class XpuAsyncLoad {
 public:
  // Task represents an asynchronous offload/reload operation.
  class Task {
   public:
    explicit Task(const Place& place);
    virtual ~Task();
    bool IsCompleted();
    void XpuSynchronize();
    void CpuSynchronize();
    void UpdateWaitChain(const phi::DeviceContext& ctx);

   private:
    Place task_place_;
    std::shared_ptr<XPUEventManager> event_manager_;
  };

  // Offload from XPU to XPUPinned memory.
  std::shared_ptr<Task> Offload(phi::DenseTensor* dst,
                                const phi::DenseTensor& src);

  // Offload a portion (with offset) from XPU to XPUPinned memory.
  std::shared_ptr<Task> OffloadWithOffset(phi::DenseTensor* dst,
                                          const phi::DenseTensor& src,
                                          size_t dst_offset,
                                          size_t src_offset,
                                          size_t offload_size);

  // Reload data from XPUPinned memory back to XPU.
  std::shared_ptr<Task> Reload(phi::DenseTensor* dst,
                               const phi::DenseTensor& src);

  // Prepare the load environment (if needed).
  void PrepareLoadEnv(const std::string& key, const Place& place);

  // Instead of using a device event’s Record/Wait, we use the XPUEventManager.
  void SyncCalcStream(const Place& place,
                      phi::XPUContext* ctx,
                      XPUEventManager* event_manager);

 private:
  std::shared_ptr<Task> CreateTask(const Place& place);

  // Map from a key (e.g. "load") to an XPUEventManager instance.
  std::unordered_map<std::string, XPUEventManager> place_to_calc_event_;
  bool is_initialized_{false};
  // std::unique_ptr<phi::XPUContext> load_ctx_;
  Place xpu_place_;
};

}  // namespace distributed
}  // namespace paddle
