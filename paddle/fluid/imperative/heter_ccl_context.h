//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_NCCL
#include "paddle/fluid/imperative/nccl_context.h"
#endif

#ifdef PADDLE_WITH_XPU_BKCL
#include "paddle/fluid/imperative/bkcl_context.h"
#endif
#include "paddle/fluid/imperative/gloo_context.h"
#include "paddle/fluid/imperative/parallel_context.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

class HeterParallelContext : public ParallelContext {
 public:
  explicit HeterParallelContext(const ParallelStrategy& strategy,
                                const int& device_id);

  ~HeterParallelContext() override = default;

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
  ParallelStrategy inter_strategy_;
  ParallelStrategy node_strategy_;
  phi::Place node_place_;
  std::shared_ptr<imperative::ParallelContext> node_parallel_ctx_{nullptr};
  std::shared_ptr<imperative::ParallelContext> inter_parallel_ctx_{nullptr};
};

}  //  namespace imperative
}  //  namespace paddle
