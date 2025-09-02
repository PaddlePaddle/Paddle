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

namespace phi {
class DenseTensor;
namespace distributed {

class GPUTask {
 public:
  GPUTask(const phi::Place& place = phi::Place(),
          const std::string& group_key = "",
          uint64_t seq = 0,
          int64_t numel = 0,
          bool sync_op = true,
          bool use_calc_stream = false,
          gpuStream_t stream = nullptr,
          CommType comm_type = CommType::UNKNOWN);
  ~GPUTask() = default;

  bool Skip();

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
  std::string group_key_;
  uint64_t seq_;
  int64_t numel_;
  bool sync_op_;
  bool use_calc_stream_;
  gpuStream_t stream_;
  CommType comm_type_;

  gpuEvent_t start_event_;
  gpuEvent_t end_event_;

  bool start_event_created_;
  bool end_event_created_;
  bool completed_;
  bool has_printed_;
  bool skip_;

 private:
  DISABLE_COPY_AND_ASSIGN(GPUTask);
};

}  // namespace distributed
}  // namespace phi
