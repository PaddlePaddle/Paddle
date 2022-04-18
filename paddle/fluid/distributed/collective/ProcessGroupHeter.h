// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/distributed/collective/ProcessGroupGloo.h"
#include "paddle/fluid/platform/device_context.h"

#ifdef PADDLE_WITH_GLOO
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#include "paddle/fluid/distributed/store/store.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/distributed/collective/NCCLTools.h"
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#include "paddle/fluid/platform/cuda_device_guard.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/distributed/collective/HCCLTools.h"
#include "paddle/fluid/distributed/collective/ProcessGroupHCCL.h"
#endif

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    (defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_ASCEND_CL))
#include "paddle/fluid/distributed/ps/service/heter_client.h"
#endif

#include "paddle/fluid/distributed/collective/Common.h"

constexpr const char* HETER_BACKEND_NAME = "HETER_BACKEND";

namespace paddle {
namespace distributed {

using Place = paddle::platform::Place;

class ProcessGroupHeter : public ProcessGroup {
 public:
  class HeterTask : public ProcessGroup::Task,
                    public std::enable_shared_from_this<HeterTask> {
   public:
    HeterTask(int rank, CommType CommType,
              const std::vector<phi::DenseTensor>&);

    bool IsCompleted();

    void SynchronizeStreams() {}

    bool Wait(std::chrono::milliseconds timeout = kWaitTimeout);

    void Synchronize() {}

    virtual ~HeterTask();
  };

  ProcessGroupHeter(const std::shared_ptr<Store>& store, int rank, int size,
                    const platform::Place& place, int gid, int local_rank,
                    int local_size, int gloo_rank, int gloo_size,
                    bool with_switch, std::string switch_endpoints);

  const std::string GetBackendName() const override {
    return std::string(HETER_BACKEND_NAME);
  }

  std::shared_ptr<ProcessGroup::Task> AllReduce(
      std::vector<phi::DenseTensor>&, std::vector<phi::DenseTensor>&,
      const AllreduceOptions& = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Task> Broadcast(
      std::vector<phi::DenseTensor>&, std::vector<phi::DenseTensor>&,
      const BroadcastOptions& = BroadcastOptions()) override;

 protected:
  virtual std::shared_ptr<ProcessGroupHeter::HeterTask> CreateTask(
      int rank, CommType opType, const std::vector<phi::DenseTensor>& inputs);

 private:
  std::shared_ptr<Store> store_;
  std::shared_ptr<ProcessGroup> inner_pg_;
  std::shared_ptr<ProcessGroupGloo> inter_pg_;

  int local_rank_;
  int local_size_;
  int gloo_rank_;
  int gloo_size_;
  bool with_switch_;
  std::string switch_endpoint_;
};

}  //  namespace distributed
}  //  namespace paddle
