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

/*
 * This file defines TensorRTSubgraphNodeMarkPass which helps to mark the ops
 * that supported by TensorRT engine.
 */

#pragma once

#include <string>
#include "paddle/fluid/inference/analysis/pass.h"
#include "paddle/fluid/inference/analysis/subgraph_splitter.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Mark the operators that TensorRT engine supports.
 */
class TensorRTSubgraphNodeMarkPass : public DataFlowGraphPass {
 public:
  using teller_t = SubGraphSplitter::NodeInsideSubgraphTeller;

  explicit TensorRTSubgraphNodeMarkPass(const teller_t& teller)
      : teller_(teller) {}

  bool Initialize(Argument* argument) override { return true; }

  // This class get a sub-graph as input and determine whether to transform this
  // sub-graph into TensorRT.
  void Run(DataFlowGraph* graph) override;

  std::string repr() const override { return "tensorrt-sub-subgraph-mark"; }
  std::string description() const override {
    return "tensorrt sub-graph mark pass";
  }

  Pass* CreateGraphvizDebugerPass() const override;
  bool Finalize() override;

 private:
  teller_t teller_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
