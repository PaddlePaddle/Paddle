// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/parallel_schedule_pass.h"
#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<Graph> ParallelSchedulePass::ApplyImpl(
    std::unique_ptr<Graph> graph) const {
  int stream_count{0};

  graph->Set(kStreamMap, new stream_map_t);
  auto& stream_map = graph->Get<stream_map_t>(kStreamMap);

  for (auto& node : GraphTraits::TS(*graph)) {
    if (node.IsOp()) {
      // Each operator has an unique stream to make them parallel.
      // The outputs share the same stream of the operator, when the operator
      // finish executes, it will generate a event.
      // When the next operator begin executes, it should sync all the events of
      // the tensor belongs.
      stream_map.emplace(&node, stream_count);
      for (auto* o : node.outputs) {
        stream_map.emplace(o, stream_count);
      }
      ++stream_count;
    }
  }
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
