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
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/collective/NCCLTools.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/device_context.h"


namespace paddle {
namespace imperative {

using CUDADeviceContext = platform::CUDADeviceContext;
using EventManager = distributed::EventManager;
using Place = platform::Place;

class OffloadScheduler {
 public:
  explicit OffloadScheduler(
      const std::vector<std::shared_ptr<imperative::VarBase>>& vars,
      const Place& src_place, const Place& dst_place, const int64_t num_streams);

  virtual ~OffloadScheduler() {}

  const std::shared_ptr<imperative::VarBase> CopyVarToGPUPlace(
    const int64_t var_index, const int64_t device_id);
  
  void WaitStreamForCopyVar(const int64_t var_index, const int64_t device_id);

 private:
  std::vector<std::shared_ptr<imperative::VarBase>> vars_;
  std::vector<std::unique_ptr<CUDADeviceContext>> contexts_;
  std::vector<EventManager> events_;

  Place src_place_;
  Place dst_place_;
  int64_t num_streams_;

};

}  // namespace imperative
}  // namespace paddle
