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

  graph->Set(kParallelMeta, new ParallelMeta);
  auto& parallel_meta = graph->Get<ParallelMeta>(kParallelMeta);

  parallel_meta.SetStreamId("feed", stream_count++);

  for (auto& node : GraphTraits::TS(*graph)) {
    if (node.IsOp()) {
      auto op_key = GenOpKey(*node.Op());
      node.Op()->SetAttr("infer_op_name", op_key);
      // Each operator has an unique stream to make them parallel.
      // The outputs share the same stream of the operator, when the operator
      // finish executes, it will generate a event.
      // When the next operator begin executes, it should sync all the events of
      // the tensor belongs.
      parallel_meta.SetStreamId(GenOpKey(*node.Op()), stream_count);
      for (auto* o : node.outputs) {
        parallel_meta.SetStreamId(o->Name(), stream_count);
      }

      // Collect event dependent relations.
      std::set<int> input_events, output_events;
      for (auto* i : node.inputs) {
        input_events.insert(parallel_meta.GetStreamId(i->Name()));
      }
      for (auto* o : node.outputs) {
        output_events.insert(parallel_meta.GetStreamId(o->Name()));
      }

      parallel_meta.SetInputDependEventIds(op_key, input_events.begin(),
                                           input_events.end());
      parallel_meta.SetOutputDependEventIds(op_key, output_events.begin(),
                                            output_events.end());

      ++stream_count;
    }

    // process the weights.
    if (node.IsVar() && node.Var()->Persistable()) {
      parallel_meta.SetStreamId(node.Name(), 0);
    }
  }

  LOG(INFO) << "get meta " << parallel_meta.StreamIds().size();

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(parallel_schedule_pass,
              paddle::framework::ir::ParallelSchedulePass)
    .RequirePassAttr(paddle::framework::ir::kParallelMeta);
