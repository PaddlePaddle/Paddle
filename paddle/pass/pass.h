// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <vector>

#include "paddle/utils/optional.h"

namespace ir {

class IrContext;
class Operation;

namespace detail {
class PassAdaptor;
}

namespace detail {

struct PassExecutionState {
  explicit PassExecutionState(ir::Operation* ir) : ir(ir), pass_failed(false) {}

  ir::Operation* ir;
  bool pass_failed;
  // TODO(liuyuanle): Add implementation of AnalysisManager and
  // PreservedAnalyses.
};

struct PassInfo {
  PassInfo(const char* name,
           uint8_t opt_level,
           const std::vector<const char* /* pass name */>& dependents = {})
      : name(name), opt_level(opt_level), dependents(dependents) {}

  // Pass name.
  const char* name;

  // opt_level=0: the basic pass which framework need.
  // opt_level=1: the fusion logical pass.
  // opt_level=2: constant fold, cse, memory optimize, etc.
  // opt_level=3: layout.
  uint8_t opt_level;

  // The list which pass depends on.
  // PassManager will check the constraint(TODO).
  std::vector<const char*> dependents;
};

}  // namespace detail

/// We can access pass only from PassManager.
class Pass {
 public:
  explicit Pass(const char* name,
                uint8_t opt_level,
                const std::vector<const char*>& dependents = {})
      : info_(name, opt_level, dependents) {}

  virtual ~Pass() = default;

  const detail::PassInfo& GetPassInfo() const { return info_; }

 protected:
  virtual void Run(ir::Operation* op) = 0;

  // TODO(liuyuanle): Add block/region judgement.
  virtual inline bool CanScheduleOn(ir::Operation* op) const { return true; }

  virtual bool Initialize(ir::IrContext* context) { return true; }

  void SignalPassFailure() { pass_state_->pass_failed = true; }

  detail::PassInfo info_;

  paddle::optional<detail::PassExecutionState> pass_state_;

  friend class PassManager;
  friend class detail::PassAdaptor;
};

}  // namespace ir
