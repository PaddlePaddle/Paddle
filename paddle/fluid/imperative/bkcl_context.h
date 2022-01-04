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

#if defined(PADDLE_WITH_XPU_BKCL)
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/imperative/parallel_context.h"
#include "xpu/bkcl.h"

namespace paddle {
namespace imperative {

class BKCLParallelContext : public ParallelContext {
 public:
  explicit BKCLParallelContext(const ParallelStrategy& strategy,
                               const platform::Place& place)
      : ParallelContext(strategy, place) {}

  ~BKCLParallelContext() override = default;

  void BcastBKCLId(std::vector<BKCLUniqueId>& bkcl_ids, int root);  // NOLINT

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
};

}  //  namespace imperative
}  //  namespace paddle

#endif
