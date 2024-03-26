//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/parallel_ssa_graph_executor.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

static std::vector<std::unique_ptr<ir::Graph>> SeparateMultiDevicesGraph(
    ir::Graph *graph, size_t place_num) {
  std::vector<std::unique_ptr<ir::Graph>> graphs;
  graphs.reserve(place_num);
  for (size_t i = 0; i < place_num; ++i) {
    ProgramDesc empty;
    graphs.emplace_back(std::make_unique<ir::Graph>(empty));
    auto &g = graphs.back();
    g->Set(kGraphVars, new GraphVars(1UL));
    g->Set(kGraphDepVars, new GraphDepVars);
    auto &stale_ops =
        graph->Get<const std::vector<OpDesc *>>(details::kStaleProgramOpDescs);
    g->Erase(details::kStaleProgramOpDescs);
    g->Set<const std::vector<OpDesc *>>(details::kStaleProgramOpDescs,
                                        new std::vector<OpDesc *>(stale_ops));
  }
  auto op_handles = ir::FilterByNodeWrapper<OpHandleBase>(*graph);

  for (auto &op : op_handles) {
    auto &dev_ctx = op->DeviceContext();
    auto &p = dev_ctx.begin()->first;
    int dev_id = p.device;  // NOLINT
    auto &dev_dummys = graphs[dev_id]->Get<GraphDepVars>(kGraphDepVars);
    graphs[dev_id]->AddNode(graph->RemoveNode(op->Node()).release());

    for (auto &var : op->Inputs()) {
      auto dummy_ptr = dynamic_cast<DummyVarHandle *>(var);
      if (dummy_ptr) {
        dev_dummys.insert(var);
        if (graph->Nodes().count(var->Node()))
          graphs[dev_id]->AddNode(graph->RemoveNode(var->Node()).release());
      }
    }
    for (auto &var : op->Outputs()) {
      auto dummy_ptr = dynamic_cast<DummyVarHandle *>(var);
      if (dummy_ptr) {
        dev_dummys.insert(var);
        if (graph->Nodes().count(var->Node()))
          graphs[dev_id]->AddNode(graph->RemoveNode(var->Node()).release());
      }
    }
  }

  for (size_t dev_id = 0; dev_id < place_num; ++dev_id) {
    auto &dev_vars = graphs[dev_id]->Get<GraphVars>(kGraphVars)[0];
    auto &origin_vars = graph->Get<GraphVars>(kGraphVars)[dev_id];
    for (auto &name_pair : origin_vars) {
      dev_vars.emplace(name_pair.first, name_pair.second);
      for (auto &version_pair : name_pair.second) {
        if (graph->Nodes().count(version_pair->Node())) {
          graphs[dev_id]->AddNode(
              graph->RemoveNode(version_pair->Node()).release());
        }
      }
    }
  }

  return graphs;
}

enum ExceptionStatus { kSuccess = 0, kEOF, kOther };

}  // namespace details
}  // namespace framework
}  // namespace paddle
