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

#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include <set>
#include "paddle/fluid/lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

using inference::analysis::Dot;

void GraphVisualizePass::Apply(std::unique_ptr<mir::SSAGraph>& graph) {
  Visualize(graph.get());
}

std::string Visualize(mir::SSAGraph* graph) {
  inference::analysis::Dot dot;

  int id = 0;
  std::set<std::string> exists_args;

  for (auto& node : graph->mutable_nodes()) {
    std::string key;
    if (node.IsArgument()) {
      key = node.AsArgument().name;
    } else {
      key = node.AsInstruct().op_type + std::to_string(id++);
    }

    if (node.IsInstruct()) {
      dot.AddNode(key, {Dot::Attr("shape", "box")});
      for (auto& x : node.inlinks) {
        auto name = x->AsArgument().name;
        if (!exists_args.count(name)) {
          dot.AddNode(name, {});
        }
        dot.AddEdge(name, key, {});
        exists_args.insert(name);
      }
      for (auto& x : node.outlinks) {
        auto name = x->AsArgument().name;
        if (!exists_args.count(name)) {
          dot.AddNode(name, {});
        }
        dot.AddEdge(key, name, {});
        exists_args.insert(name);
      }
    }
  }

  auto res = dot.Build();
  LOG(INFO) << "dot:\n" << res;
  return res;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(graph_visualze, paddle::lite::mir::GraphVisualizePass);
