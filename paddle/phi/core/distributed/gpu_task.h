// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/platform/resource_pool.h"

namespace phi {
class DenseTensor;
namespace distributed {

class GPUTask {
 public:
  GPUTask(const phi::Place& place,
          gpuStream_t stream = nullptr,
          const std::string& label = "");
  ~GPUTask() = default;

  void StartRecord();
  void EndRecord();
  bool CudaEventQuery(gpuEvent_t event);
  bool IsCompleted();
  void ClearRecord();

  bool HasPrinted();
  void SetPrint();
  std::string GetTraceMsg(gpuEvent_t event);

 private:
  phi::Place place_;
  gpuStream_t stream_;
  std::string label_;

  gpuEvent_t start_event_;
  gpuEvent_t end_event_;
  // std::shared_ptr<gpuEvent_t> start_event_;
  // std::shared_ptr<gpuEvent_t> end_event_;

  bool start_event_created_;
  bool end_event_created_;
  bool completed_;
  bool has_printed_;

 private:
  DISABLE_COPY_AND_ASSIGN(GPUTask);
};

class CudaEventResourcePool {
 public:
  std::shared_ptr<gpuEvent_t> New(int dev_idx);

  static CudaEventResourcePool& Instance();

 private:
  CudaEventResourcePool();

  DISABLE_COPY_AND_ASSIGN(CudaEventResourcePool);

 private:
  std::vector<std::shared_ptr<paddle::platform::ResourcePool<gpuEvent_t>>>
      pool_;
};

}  // namespace distributed
}  // namespace phi
