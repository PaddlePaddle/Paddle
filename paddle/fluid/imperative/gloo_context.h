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

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/parallel_context.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace imperative {

class GLOOParallelContext : public ParallelContext {
 public:
  explicit GLOOParallelContext(const ParallelStrategy& strategy,
                               const phi::Place& place)
      : ParallelContext(strategy, place) {}

  ~GLOOParallelContext() override = default;

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
  void AllReduce(const phi::DenseTensor& src, phi::DenseTensor* dst);
  void AllReduce(const phi::SelectedRows& src, phi::SelectedRows* dst);

 private:
  std::unique_ptr<phi::CPUContext> device_;
};

}  //  namespace imperative
}  //  namespace paddle
