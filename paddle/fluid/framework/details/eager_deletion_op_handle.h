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
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"

namespace paddle {
namespace framework {
class GarbageCollector;
class Scope;

namespace ir {
class Node;
}  // namespace ir

namespace ir {
class MemOptVarInfo;
}  // namespace ir

namespace details {

class EagerDeletionOpHandle : public OpHandleBase {
 public:
  EagerDeletionOpHandle(ir::Node *node,
                        Scope *scope,
                        size_t scope_idx,
                        const phi::Place &place,
                        const std::unordered_set<ir::MemOptVarInfo *> &vars,
                        GarbageCollector *gc);

  ~EagerDeletionOpHandle();

  std::string Name() const override;

  /**
   * Currently, EagerDeletionOpHandle has the highest priority.
   * This priority settings speed up gc 15% in Transformer
   * V100 8-GPU model.
   */
  Priority GetPriority() const override { return kHighest; }

  size_t GetScopeIdx() const { return scope_idx_; }

  std::vector<std::string> VarsToDelete() const;

 protected:
  void RunImpl() override;

  void InitCUDA() override;

  std::vector<Scope *> GetLocalScopes() override { return {scope_}; }

 private:
  void ClearGarbages(std::deque<std::shared_ptr<memory::Allocation>> *garbages);

  void CallOnce();

  Scope *scope_;
  size_t scope_idx_;
  phi::Place place_;
  std::vector<ir::MemOptVarInfo *> var_infos_;  // not own
  GarbageCollector *gc_;                        // not own
  std::vector<Variable *> vars_;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  phi::GPUContext *dev_ctx_{nullptr};
  gpuEvent_t event_{nullptr};
#endif
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
