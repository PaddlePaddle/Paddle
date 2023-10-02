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
#include <string>
#include <vector>

#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/pass/analysis_manager.h"
#include "paddle/pir/pass/pass_registry.h"

namespace pir {

class IrContext;
class Operation;

namespace detail {
class PassAdaptor;
}

namespace detail {

struct PassExecutionState {
  explicit PassExecutionState(Operation* ir, const AnalysisManager& am)
      : ir(ir), pass_failed(false), am(am) {}

  // The IR currently being processed by pass.
  Operation* ir;

  bool pass_failed;
  AnalysisManager am;
  PreservedAnalyses preserved_analyses;
};

struct PassInfo {
  PassInfo(const std::string& name,
           uint8_t opt_level,
           const std::vector<std::string /* pass name */>& dependents = {})
      : name(name), opt_level(opt_level), dependents(dependents) {}

  // Pass name.
  std::string name;

  // opt_level=0: the basic pass which framework need.
  // opt_level=1: constant fold, cse, memory optimize, etc.
  // opt_level=2: the fusion logical pass.
  // opt_level=3: layout, etc.
  uint8_t opt_level;

  // The list which pass depends on.
  // PassManager will check the constraint(TODO).
  std::vector<std::string> dependents;
};

}  // namespace detail

/// We can access pass only from PassManager.
class IR_API Pass {
 public:
  explicit Pass(const std::string& name,
                uint8_t opt_level,
                const std::vector<std::string>& dependents = {})
      : pass_info_(name, opt_level, dependents) {}

  virtual ~Pass();

  const std::string& name() const { return pass_info().name; }

  const detail::PassInfo& pass_info() const { return pass_info_; }

 protected:
  virtual void Run(Operation* op) = 0;

  virtual inline bool CanApplyOn(Operation* op) const;

  virtual bool Initialize(IrContext* context) { return true; }

  AnalysisManager analysis_manager() { return pass_state().am; }

  detail::PassExecutionState& pass_state() {
    IR_ENFORCE(pass_state_.has_value() == true, "pass state has no value");
    return *pass_state_;
  }

  void SignalPassFailure() { pass_state().pass_failed = true; }

 private:
  detail::PassInfo pass_info_;

  std::optional<detail::PassExecutionState> pass_state_;

  friend class PassManager;
  friend class detail::PassAdaptor;
};

}  // namespace pir
