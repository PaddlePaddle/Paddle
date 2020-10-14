// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/batch_norm_act_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void FuseBatchNormActOneDNNPass::ApplyImpl(Graph *graph) const {
  std::string act_type("relu");
  FuseBatchNormAct(graph, act_type);
}

void FuseBatchNormActOneDNNPass::FuseBatchNormAct(
    Graph *graph, const std::string &act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument(
                 "The input graph of "
                 "FuseBatchNormActOneDNNPass should not be nullptr."));
  FusePassBase::Init("bn_act", graph);

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("bn_act/x")
                ->AsInput()
                ->assert_is_op_input("batch_norm", "X");
  patterns::BatchNormActOneDNN bn_act_pattern(gpd.mutable_pattern(), "bn_act");
  bn_act_pattern(x, act_type);

  int found_bn_act_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse BatchNorm with ReLU activation op.";
    // BN output
    GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, bn_act_pattern);
    // ACT output
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, bn_act_pattern);
    // ops
    GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, bn_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, act, bn_act_pattern);

    batch_norm->Op()->SetAttr("fuse_with_relu", true);
    batch_norm->Op()->SetOutput("Y", {act_out->Name()});

    IR_OP_VAR_LINK(batch_norm, act_out);
    GraphSafeRemoveNodes(g, {act, bn_out});
    found_bn_act_count++;
  };

  gpd(graph, handler);
  AddStatis(found_bn_act_count);
  PrettyLogDetail("---    fused %d batch norm with relu activation",
                  found_bn_act_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(batch_norm_act_fuse_pass,
              paddle::framework::ir::FuseBatchNormActOneDNNPass);
REGISTER_PASS_CAPABILITY(batch_norm_act_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("batch_norm", 0)
            .EQ("relu", 0));
