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

#include <map>
#include <vector>

#include "paddle/fluid/framework/new_executor/new_executor_defs.h"

PD_DECLARE_bool(new_executor_sequential_run);

namespace paddle {
namespace framework {
class InstructionBase;
namespace interpreter {

// DependencyBuilder provides some dependency adding function to handle the
// dependency that cannot be explicitly expresed by a Program. It is a
// compromise of the incomplete expression ability of the Program. Do not add
// too many functions here at will, that will bring great burden to the
// Interpretercore.

class DependencyBuilder {
 public:
  DependencyBuilder();

  // build op dependencies and return the mapping from op to its downstream-op
  // set
  const std::map<size_t, std::set<size_t>>& Build(
      const std::vector<Instruction>& instructions);

  std::tuple<std::shared_ptr<std::map<size_t, std::set<size_t>>>,
             std::shared_ptr<std::vector<std::vector<bool>>>>
  GetDependency() const;

  const std::map<size_t, std::set<size_t>>& OpDownstreamMap() const;

  bool OpHappensBefore(size_t prior_op_idx, size_t posterior_op_idx) const {
    PADDLE_ENFORCE_GE(
        op_happens_before_->size(),
        0,
        phi::errors::Unavailable("op_happen_before is not yet built"));
    return op_happens_before_->at(prior_op_idx).at(posterior_op_idx);
  }

  void ShareDependencyFrom(const DependencyBuilder& src);

 protected:
  void AddDependencyForCoalesceTensorOp();
  void AddDependencyForCommunicationOp();
  void AddDependencyForRandomOp();
  void AddDependencyForReadOp();
  void AddDependencyForSequentialRun();

  void AddDownstreamOp(size_t prior_op_idx, size_t posterior_op_idx);

  void BuildDownstreamMap();

  void ShrinkDownstreamMap();

  void UpdateVarMinRwOp(
      const std::map<size_t, std::set<size_t>>& op2dependences,
      std::map<size_t, std::list<size_t>>* var2min_rw_op,
      size_t cur_op,
      size_t rw_var);

  bool is_build_;

  size_t op_num_;

  // ops_behind_ is the adjacency list about op to its posterior-ops, that is to
  // say, op_behind_[i] == {a, b, c} means op[a], op[b] and op[c] depend on
  // op[i] directly or indirectly. ops_before_ is the revered adjacency list of
  // ops_behind_.
  std::vector<std::vector<size_t>> ops_before_;
  std::vector<std::vector<size_t>> ops_behind_;

  // op_downstream_map_ is the mapping from op to its downstream-op set, that is
  // to say, (*op_downstream_map_)[i] == {a, b, c} means op[a], op[b] and op[c]
  // depend on op[i] directly.
  std::shared_ptr<std::map<size_t, std::set<size_t>>> op_downstream_map_;

  // op_happens_before_ is a matrix form of ops_before_ and ops_behind_, it is
  // used to speed up the query.
  std::shared_ptr<std::vector<std::vector<bool>>> op_happens_before_;

 private:
  const std::vector<Instruction>* instructions_;  // not_own
};

/// ======================== ///
///        For new ir        ///
/// ======================== ///
class NewIrDependencyBuilder : public DependencyBuilder {
 public:
  NewIrDependencyBuilder();

  // build op dependencies and return the mapping from op to its downstream-op
  // set
  const std::map<size_t, std::set<size_t>>& Build(
      std::vector<paddle::framework::InstructionBase*> instructions);

  void BuildDownstreamMap();

  void ShareDependencyFrom(const NewIrDependencyBuilder& src);

 private:
  std::vector<paddle::framework::InstructionBase*> instructions_;  // not_owned
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
