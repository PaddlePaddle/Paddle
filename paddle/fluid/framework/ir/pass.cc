/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/pass.h"

#include <memory>
#include <utility>

#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {

Graph* Pass::Apply(Graph* graph) const {
  CheckPrevPass();
  PADDLE_ENFORCE(graph, "graph passed to Pass::Apply() cannot be empty.");
  for (const std::string& attr : required_pass_attrs_) {
    PADDLE_ENFORCE(attrs_.find(attr) != attrs_.end(),
                   "Required pass atrribute %s not set.", attr);
  }
  for (const std::string& attr : required_graph_attrs_) {
    PADDLE_ENFORCE(graph->Has(attr), "Required graph atrribute %s not set.",
                   attr);
  }
  ApplyImpl(graph);
  // TODO(panyx0718): Add more verifications.
  PADDLE_ENFORCE(!HasCircle(*graph),
                 "Illegal Pass %s. Generated graph shouldn't have cycle.",
                 Type());
  PADDLE_ENFORCE(VarDescIsConsistency(*graph),
                 "The VarDescs of persistable variable are not consistency.");
  applied_ = true;
  if (!graph->Has(kPassRecorder)) {
    graph->Set<PassRecorder>(kPassRecorder, new PassRecorder);
  }
  graph->Get<PassRecorder>(kPassRecorder).insert(Type());
  return graph;
}

PassRegistry& PassRegistry::Instance() {
  static PassRegistry g_pass_info_map;
  return g_pass_info_map;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
