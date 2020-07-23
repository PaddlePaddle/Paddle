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
#include "paddle/fluid/platform/device_context.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace framework {
namespace ir {

Graph* Pass::Apply(Graph* graph) const {
  CheckPrevPass();
  PADDLE_ENFORCE(graph, "graph passed to Pass::Apply() cannot be empty.");
  for (const std::string& attr : required_pass_attrs_) {
    PADDLE_ENFORCE_NE(
        attrs_.find(attr), attrs_.end(),
        platform::errors::InvalidArgument(
            "Required atrribute %s for pass < %s > is not set.", attr, Type()));
  }
  for (const std::string& attr : required_graph_attrs_) {
    PADDLE_ENFORCE_EQ(graph->Has(attr), true,
                      platform::errors::InvalidArgument(
                          "Required atrribute %s for graph is not set.", attr));
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
#ifdef PADDLE_WITH_MKLDNN
  // Clear mkl-dnn cache,
  // Passes can change params, tensors, so caching need to be discarded
  ClearMKLDNNCache(paddle::platform::CPUPlace());
#endif
  return graph;
}

PassRegistry& PassRegistry::Instance() {
  static PassRegistry g_pass_info_map;
  return g_pass_info_map;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
