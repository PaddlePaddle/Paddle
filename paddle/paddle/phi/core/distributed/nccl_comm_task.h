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

#include "paddle/common/macros.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/distributed/comm_task.h"
#include "paddle/phi/core/distributed/utils.h"

#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#else
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi {
class DenseTensor;
namespace distributed {

static int64_t DefaultTimeout = 30 * 60 * 1000;

class NCCLCommTask : public CommTask {
 public:
  NCCLCommTask(const phi::Place& place = phi::Place(),
               const std::string& group_key = "",
               int rank = -1,
               int size = 0,
               int gid = 0,
               uint64_t seq = 0,
               int64_t numel = 0,
               bool sync_op = true,
               bool use_calc_stream = false,
               ncclComm_t = nullptr,
               gpuStream_t = nullptr,
               CommType comm_type = CommType::UNKNOWN,
               int64_t timeout = DefaultTimeout);
  ~NCCLCommTask() override = default;

  // check whether the nccl kernel started
  bool IsStarted() override;
  bool IsTimeout() override;
  bool IsCompleted() override;
  void SetUpdated(bool updated) override;
  bool IsUpdated() override;

  std::string GetTraceMsg() override;
  std::string GetCommErrors() override;
  void AbortComm() override;

  void StartRecord() override;
  void EndRecord() override;
  void ClearRecord() override;

  bool CudaEventQuery(gpuEvent_t event);

 protected:
  std::mutex mutex_;
  std::chrono::milliseconds timeout_;

#ifdef PADDLE_WITH_CUDA
  unsigned int cuda_event_flags_ = cudaEventDisableTiming;
#else  // PADDLE_WITH_HIP
  unsigned int hip_event_flags_ = hipEventDisableTiming;
#endif

  bool sync_op_;
  bool use_calc_stream_;

  bool start_event_created_;
  bool end_event_created_;
  gpuEvent_t nccl_start_event_;
  gpuEvent_t nccl_end_event_;

  std::string comm_error_;

 private:
  DISABLE_COPY_AND_ASSIGN(NCCLCommTask);
};

}  // namespace distributed
}  // namespace phi
