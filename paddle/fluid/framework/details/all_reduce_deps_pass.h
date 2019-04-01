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
#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

// TODO(gongwb): overlap allreduce with backward computation.
class AllReduceDepsPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  std::vector<OpHandleBase *> GetSortedOpFromGraph(
      const ir::Graph &graph) const;

  std::map<int, std::vector<std::string>> GetSoredGradientsFromStaleProgram(
      const ir::Graph &graph) const;

  void PrintVlog(
      const ir::Graph &graph,
      const std::vector<AllReduceOpHandle *> &all_reduce_op_handles) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
