// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {
namespace details {

using ReferenceCountMap = std::unordered_map<std::string, int>;
using AtomicReferenceCountMap =
    std::unordered_map<std::string, std::atomic<int>>;
using DeviceReferenceCountMap =
    std::unordered_map<int, std::unique_ptr<ReferenceCountMap>>;
using AtomicDeviceReferenceCountMap =
    std::unordered_map<int, std::unique_ptr<AtomicReferenceCountMap>>;
using DeviceGarbageCollectorMap =
    std::unordered_map<int, std::unique_ptr<GarbageCollector<Tensor>>>;

class ReferenceCountOpHandle : public OpHandleBase {
 public:
  ReferenceCountOpHandle(ir::Node *node, const Scope *scope,
                         const platform::CUDAPlace &place,
                         const std::vector<std::string> &var_names,
                         GarbageCollector<Tensor> *gc,
                         AtomicReferenceCountMap *ref_cnts);

  ~ReferenceCountOpHandle();

  std::string Name() const override;

  void AddVar(const std::string &name);

 protected:
  void RunImpl() override;

 private:
  void ClearTensors(const std::vector<Tensor *> &tensors);

  const Scope *scope_;
  platform::CUDADeviceContext *dev_ctx_;
  std::unordered_map<std::string, int> var_names_;
  GarbageCollector<Tensor> *gc_;       // not own
  AtomicReferenceCountMap *ref_cnts_;  // not own
  cudaEvent_t event_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
