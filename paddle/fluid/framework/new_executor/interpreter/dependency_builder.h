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

namespace paddle {
namespace framework {
namespace interpreter {

// DependencyBuilder provides some dependency adding function to handle the
// dependency that cannot be explicitly expresed by a Program. It is a
// compromise of the incomplete expression ability of the Program. Do not add
// too many functions here at will, that will bring great burden to the
// Interpretercore.

class DependencyBuilder {
 public:
  DependencyBuilder() : is_build_(false), instructions_(nullptr) {}

  // build op dependencies and return the mapping from op to its downstream-op
  // set
<<<<<<< HEAD
  const std::map<size_t, std::set<size_t>>& Build(
      const std::vector<Instruction>& instructions);

  const std::map<size_t, std::set<size_t>>& OpDownstreamMap() const;

  bool OpHappensBefore(size_t prior_op_idx, size_t posterior_op_idx) const;
=======
  const std::map<int, std::set<int>>& Build(
      const std::vector<Instruction>& instructions);

  bool OpHappensBefore(int prior_op_idx, int posterior_op_idx);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

 private:
  void AddDependencyForCoalesceTensorOp();
  void AddDependencyForCommunicationOp();
  void AddDependencyForRandomOp();
  void AddDependencyForReadOp();
  void AddDependencyForSequentialRun();

<<<<<<< HEAD
  void AddDownstreamOp(size_t prior_op_idx, size_t posterior_op_idx);
=======
  void AddDownstreamOp(int prior_op_idx, int posterior_op_idx);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  void BuildDownstreamMap();

  void BuildOpHappensBefore();

  void ShrinkDownstreamMap();

  bool is_build_;
  const std::vector<Instruction>* instructions_;  // not_own
  size_t op_num_;

  // op_happens_before_[i][j] == true means op[i] happens before op[j]
  std::vector<std::vector<bool>> op_happens_before_;

  // op_downstream_map_ is the mapping from op to its downstream-op set, that is
  // to say, op_downstream_map_[i] == {a, b, c} means op[a], op[b] and op[c]
  // should be dispatched after op[i]
<<<<<<< HEAD
  std::map<size_t, std::set<size_t>> op_downstream_map_;
=======
  std::map<int, std::set<int>> op_downstream_map_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
