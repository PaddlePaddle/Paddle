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

#include "paddle/fluid/framework/new_executor/interpreter/add_dependency.h"

#include <queue>

namespace paddle {
namespace framework {
namespace interpreter {

void AddDownstreamOp(int prior_op_idx,
                     int posterior_op_idx,
                     std::map<int, std::list<int>>* op_downstream_map,
                     const std::vector<std::vector<bool>>* op_happens_before) {
  if (op_downstream_map->find(prior_op_idx) == op_downstream_map->end()) {
    op_downstream_map->emplace(std::make_pair(prior_op_idx, std::list<int>()));
  } else {
    if (op_happens_before != nullptr) {
      for (int op_idx : op_downstream_map->at(prior_op_idx)) {
        if (op_happens_before->at(op_idx).at(posterior_op_idx)) {
          VLOG(7) << "Find dependencies " << prior_op_idx << "->" << op_idx
                  << "->" << posterior_op_idx << ", skip adding "
                  << prior_op_idx << "->" << posterior_op_idx;
          return;
        }
      }
    }
  }

  op_downstream_map->at(prior_op_idx).push_back(posterior_op_idx);
}

// check whether exists prior_op -> ... -> posterior_op to avoid building loops
bool IsDependency(int prior_op_idx,
                  int posterior_op_idx,
                  const std::map<int, std::list<int>>& downstream_map) {
  std::queue<int> q;
  q.push(prior_op_idx);

  while (!q.empty()) {
    int op_idx = q.front();
    q.pop();

    auto it = downstream_map.find(op_idx);
    if (it != downstream_map.end()) {
      for (int downstream_op_idx : it->second) {
        if (downstream_op_idx == posterior_op_idx) {
          return true;
        }

        // no need for double enqueue checking since DAG is assumed
        q.push(downstream_op_idx);
      }
    }
  }

  return false;
}

void AddDependencyForReadOp(
    const std::vector<Instruction>& vec_instruction,
    std::map<int, std::list<int>>* downstream_map,
    const std::vector<std::vector<bool>>* op_happens_before) {
  size_t op_num = vec_instruction.size();
  std::vector<bool> is_startup_ops(op_num, true);
  for (size_t op_idx = 0; op_idx < op_num; ++op_idx) {
    auto it = downstream_map->find(op_idx);
    if (it != downstream_map->end()) {
      for (size_t downstream_op_idx : it->second) {
        is_startup_ops[downstream_op_idx] = false;
      }
    }
  }

  std::vector<size_t> read_ops;
  std::vector<size_t> startup_ops;
  for (size_t op_idx = 0; op_idx < op_num; ++op_idx) {
    if (vec_instruction[op_idx].OpBase()->Type() == "read") {
      read_ops.push_back(op_idx);
    }

    if (is_startup_ops[op_idx]) {
      startup_ops.push_back(op_idx);
    }
  }

  for (size_t read_op_idx : read_ops) {
    for (size_t downstream_op_idx : startup_ops) {
      if (read_op_idx != downstream_op_idx &&
          !IsDependency(downstream_op_idx, read_op_idx, *downstream_map))
        AddDownstreamOp(
            read_op_idx, downstream_op_idx, downstream_map, op_happens_before);
      VLOG(4) << "Add depend from "
              << vec_instruction[read_op_idx].OpBase()->Type() << "("
              << read_op_idx << ") to "
              << vec_instruction[downstream_op_idx].OpBase()->Type() << "("
              << downstream_op_idx << ")";
    }
  }
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
