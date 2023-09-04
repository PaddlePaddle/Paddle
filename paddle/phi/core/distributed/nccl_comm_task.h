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

#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/distributed/comm_task.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/macros.h"

#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#else
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi {
class DenseTensor;
namespace distributed {
class Store;

static constexpr std::chrono::milliseconds DefaultTimeout =
    std::chrono::seconds(300);
static constexpr std::chrono::milliseconds NoTimeout =
    std::chrono::milliseconds::zero();

class NCCLCommTask : public CommTask {
 public:
  NCCLCommTask(const phi::Place& place = phi::Place(),
               int rank = -1,
               int size = 0,
               uint64_t seq = 0,
               int64_t numel = 0,
               bool sync_op = true,
               bool use_calc_stream = false,
               ncclComm_t = nullptr,
               gpuStream_t = nullptr,
               CommType comm_type = CommType::UNKNOWN);
  ~NCCLCommTask() = default;

  // check whether the nccl kernel started
  bool IsStarted() override;
  bool IsTimeout() override;
  bool IsCompleted() override;
  bool IsSuccess() override;

  void SetException(std::exception_ptr exception) override;
  void CheckAndSetException() override;
  std::exception_ptr CheckCommErrors() override;
  std::exception_ptr GetException() override;
  void AbortComm() override;

  void StartRecord();
  void EndRecord(phi::gpuStream_t stream);

  bool CudaEventQuery(cudaEvent_t event);

  std::shared_ptr<Store> store_;

 protected:
  // task timeout threshold
  std::chrono::milliseconds timeout_;
  // task started time

  unsigned int cuda_event_flags_ = cudaEventDisableTiming;

  ncclResult_t nccl_async_err_;

  uint64_t seq_;
  int64_t numel_;
  bool sync_op_;
  bool use_calc_stream_;

  ncclComm_t nccl_comm_;
  gpuStream_t nccl_stream_;

  bool start_event_created_;
  bool end_event_created_;
  cudaEvent_t nccl_start_event_;
  cudaEvent_t nccl_end_event_;

  std::string comm_failure_reason_;
  ncclResult_t nccl_async_error_;

 private:
  DISABLE_COPY_AND_ASSIGN(NCCLCommTask);
};

}  // namespace distributed
}  // namespace phi
