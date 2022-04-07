//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/gpu_resource_pool.h"
#endif

#ifdef PADDLE_WITH_NCCL
#include "paddle/fluid/platform/dynload/nccl.h"
#endif

#ifdef PADDLE_WITH_RCCL
#include "paddle/fluid/platform/dynload/rccl.h"
#endif

#include "paddle/fluid/imperative/parallel_context.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
class NCCLParallelContext : public ParallelContext {
 public:
  explicit NCCLParallelContext(const ParallelStrategy& strategy,
                               const platform::Place& place)
      : ParallelContext(strategy, place) {}

  ~NCCLParallelContext() override = default;

  void BcastNCCLId(std::vector<ncclUniqueId>& nccl_ids, int root,  // NOLINT
                   int server_fd);

  void Init() override;

  void InitWithRingID(int ring_id) override;

  void AllReduceByStream(const framework::Variable& src,
                         framework::Variable* dst, int ring_id,
                         bool use_calc_stream) override;

  void Broadcast(framework::Variable* src, int ring_id) override;

  paddle::platform::DeviceContext* GetDeviceContext(int ring_id) override;

  void WaitCompute(int ring_id) override;

  void WaitComm(int ring_id) override;

  void SynchronizeCompute() override;

 private:
  // used for comm wait compute, compute_stream-->event-->comm_stream[ring_id]
  std::vector<std::shared_ptr<platform::CudaEventObject>> compute_events_;

  // used for compute wait comm, comm_stream[ring_id]-->event-->compute_stream
  std::vector<std::shared_ptr<platform::CudaEventObject>> comm_events_;
};
#endif

}  //  namespace imperative
}  //  namespace paddle
