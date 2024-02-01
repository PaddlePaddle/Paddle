// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <functional>
#include <limits>
#include <vector>

#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_compare.h"

namespace cinn {
namespace auto_schedule {

struct _SearchState_;
class AutoGenRule;

//! Shared Wrapper for _SearchState_
class SearchState : public cinn::common::Shared<_SearchState_> {
 public:
  SearchState() = default;
  // create a new SearchState
  explicit SearchState(ir::IRSchedule ir_sch,
                       float cost = NOT_INIT_COST,
                       const std::vector<AutoGenRule*>& rules = {});

  // Constant standing for a cost not being initialized
  static constexpr float NOT_INIT_COST = std::numeric_limits<float>::max();
  // compare function for two states
  friend bool operator<(const SearchState& left, const SearchState& right);

  // Deep copy a SearchState
  SearchState Copy() const;
};

//! Class to store immediate states during search
struct _SearchState_ : public cinn::common::Object {
  // IRSchedule contains ir::ModuleExpr and trace scheduling process
  ir::IRSchedule ir_schedule;
  // Cost model predicted cost
  float predicted_cost;
  // The rules that can be applied to the IRSchedule at this state.
  std::vector<AutoGenRule*> applicable_rules;

  // return detail string of content for debug;
  std::string DebugString() const;

  const char* type_info() const override { return __type_info__; }
  static constexpr char* __type_info__ = "auto_schedule_state";
};

// SearchStateHash hash functor that visits every AST node and combine their
// hash of node_type in dfs order
struct SearchStateHash {
  size_t operator()(const SearchState& s) const;
};

// SearchStateHash equal functor, use ir::ir_utils::IrEqualVisitor to compare
// their AST struct and fields
struct SearchStateEqual {
  bool operator()(const SearchState& lhs, const SearchState& rhs) const;
};

/*!
 * \brief concatenate debug strings of all states with additional info
 * \param title head of the result string
 * \param states SearchState array to be debugged
 * \param verbose whether to enable more verbose debug info
 * \return the concatenated debug string
 */
std::string JoinStatesDebugString(const std::string& title,
                                  const std::vector<SearchState>& states,
                                  bool verbose = false);

}  // namespace auto_schedule
}  // namespace cinn
