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

#include <any>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/pir/pass/analysis_manager.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"

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
  // opt_level=4: the radical optimization, maybe affect precision, etc.
  uint8_t opt_level;

  // The list which pass depends on.
  // PassManager will check the constraint(TODO).
  std::vector<std::string> dependents;
};

}  // namespace detail

static const char kParamScopeAttr[] = "__param_scope__";
static const char kPlaceAttr[] = "__place__";

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

  // Get a reference to the attributed previously set.
  template <typename AttrType>
  AttrType& Get(const std::string& attr_name) const {
    IR_ENFORCE(attrs_.find(attr_name) != attrs_.end(),
               "Attribute %s not registered for pass.",
               attr_name);
    try {
      return *std::any_cast<AttrType*>(attrs_.at(attr_name));
    } catch (std::bad_any_cast&) {
      auto TypeToString = [](const std::type_info& info) -> std::string {
        if (std::type_index(info) == std::type_index(typeid(bool*))) {
          return "bool";
        } else if (std::type_index(info) == std::type_index(typeid(int*))) {
          return "int";
        } else if (std::type_index(info) ==
                   std::type_index(typeid(const int*))) {
          return "const int";
        } else if (std::type_index(info) ==
                   std::type_index(typeid(std::string*))) {
          return "std::string";
        }
        return info.name();
      };

      IR_THROW("Invalid type for attribute %s, expected: %s, actual: %s.",
               attr_name,
               TypeToString(typeid(AttrType*)),
               TypeToString(attrs_.at(attr_name).type()));
    }
  }

  bool Has(const std::string& attr_name) const {
    return attrs_.count(attr_name) > 0;
  }

  void Erase(const std::string& attr_name) {
    if (!Has(attr_name)) {
      return;
    }
    if (attr_dels_.find(attr_name) != attr_dels_.end()) {
      attr_dels_[attr_name]();
      attr_dels_.erase(attr_name);
    }
    attrs_.erase(attr_name);
  }

  // Set a pointer to the attribute. Pass takes ownership of the attribute.
  template <typename AttrType>
  void Set(const std::string& attr_name, AttrType* attr) {
    VLOG(3) << "Setting the attribute " << attr_name << " for the pass "
            << name();
    if (Has(attr_name)) {
      Erase(attr_name);
    }
    attrs_[attr_name] = attr;
    attr_dels_[attr_name] = [attr, attr_name]() {
      VLOG(8) << "deleting " << attr_name;
      delete attr;
    };
  }

  // Set a pointer to the attribute. Pass doesn't take ownership. Caller
  // should delete the attribute.
  template <typename AttrType>
  void SetNotOwned(const std::string& attr_name, AttrType* attr) {
    VLOG(3) << "Setting the attribute " << attr_name << " for the " << name();
    IR_ENFORCE(
        !Has(attr_name), "Attribute %s already set in the pass.", attr_name);
    attrs_[attr_name] = attr;
  }

 protected:
  virtual void Run(Operation* op) = 0;

  virtual bool CanApplyOn(Operation* op) const;

  virtual bool Initialize(IrContext* context) { return true; }

  void AddStatistics(int64_t match_count) {
    Set<int64_t>("__match_count__", new int64_t{match_count});
  }

  void AddStatistics(int64_t match_count, int64_t all_count) {
    Set<int64_t>("__match_count__", new int64_t{match_count});
    Set<int64_t>("__all_count__", new int64_t{all_count});
  }

  void AddStatistics(const std::string& custom_log) {
    Set<std::string>("__custom_log__", new std::string{custom_log});
  }

  AnalysisManager analysis_manager() { return pass_state().am; }

  detail::PassExecutionState& pass_state();

  void SignalPassFailure() { pass_state().pass_failed = true; }

 private:
  detail::PassInfo pass_info_;

  std::optional<detail::PassExecutionState> pass_state_;

  friend class PassManager;
  friend class detail::PassAdaptor;

  std::unordered_map<std::string, std::any> attrs_;
  std::unordered_map<std::string, std::function<void(void)>> attr_dels_;
};

class IR_API PatternRewritePass : public Pass {
 public:
  PatternRewritePass(const std::string& name,
                     uint8_t opt_level,
                     const std::vector<std::string>& dependents = {})
      : Pass(name, opt_level, dependents) {}

 protected:
  virtual RewritePatternSet InitializePatterns(IrContext* context) = 0;

  bool Initialize(IrContext* context) final;

  void Run(Operation* op) override;

 private:
  FrozenRewritePatternSet patterns_;
};

}  // namespace pir
