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

#include <deque>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/reference_count_pass_helper.h"

namespace paddle {
namespace framework {
class Scope;

namespace details {

class EagerDeletionOpHandle : public OpHandleBase {
 public:
  EagerDeletionOpHandle(ir::Node *node, const Scope *scope,
                        const platform::Place &place,
                        const std::unordered_set<std::string> &var_names,
                        GarbageCollector *gc,
                        AtomicReferenceCountMap *ref_cnts);

  ~EagerDeletionOpHandle();

  std::string Name() const override;

  Priority GetPriority() const override { return kHighest; }

 protected:
  void RunImpl() override;

 private:
  void ClearGarbages(std::deque<std::shared_ptr<memory::Allocation>> *garbages);

  const Scope *scope_;
  std::vector<std::string> var_names_;
  GarbageCollector *gc_;               // not own
  AtomicReferenceCountMap *ref_cnts_;  // not own
#ifdef PADDLE_WITH_CUDA
  platform::CUDADeviceContext *dev_ctx_{nullptr};
  cudaEvent_t event_{nullptr};
#endif
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
