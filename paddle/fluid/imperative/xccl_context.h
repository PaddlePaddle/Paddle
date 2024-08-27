//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/parallel_context.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
class XCCLParallelContext : public ParallelContext {
 public:
  explicit XCCLParallelContext(const ParallelStrategy& strategy,
                               const phi::Place& place)
      : ParallelContext(strategy, place) {}

  ~XCCLParallelContext() override = default;

  void BcastXCCLId(std::vector<phi::ccl::CCLRootId>& xccl_ids,  // NOLINT
                   int root,
                   int server_fd);

  void Init() override;

  void InitWithRingID(int ring_id) override;

  void AllReduceByStream(const framework::Variable& src,
                         framework::Variable* dst,
                         int ring_id,
                         bool use_calc_stream) override;

  void Broadcast(framework::Variable* src, int ring_id) override;

  phi::DeviceContext* GetDeviceContext(int ring_id) override;

  void WaitCompute(int ring_id) override;

  void WaitComm(int ring_id) override;

  void SynchronizeCompute() override;

 private:
  // used for comm wait compute, compute_stream-->event-->comm_stream[ring_id]
  std::vector<std::shared_ptr<phi::event::Event>> compute_events_;

  // used for compute wait comm, comm_stream[ring_id]-->event-->compute_stream
  std::vector<std::shared_ptr<phi::event::Event>> comm_events_;
};
#endif
}  //  namespace imperative
}  //  namespace paddle
