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

#pragma once

#include "paddle/fluid/inference/analysis/node.h"
#include "paddle/fluid/inference/analysis/pass.h"
#include "paddle/fluid/inference/analysis/subgraph_splitter.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Parse the graph and replace TensorRT supported nodes with SubGraphNode
 */
class TensorRTSubGraphPass : public DataFlowGraphPass {
 public:
  // Tell whether to transform a sub-graph into TensorRT.
  using NodeInsideSubgraphTeller = SubGraphFuse::NodeInsideSubgraphTeller;

  TensorRTSubGraphPass(const NodeInsideSubgraphTeller& teller);

  bool Initialize(Argument* argument) override { return true; }

  // This class get a sub-graph as input and determine whether to transform this
  // sub-graph into TensorRT.
  void Run(DataFlowGraph* graph) override;

 private:
  NodeInsideSubgraphTeller node_inside_subgraph_teller_;
};

}  // namespace analysis
}  // namespace inference
}  // paddle
