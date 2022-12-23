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

#include "paddle/fluid/framework/ir/mkldnn/multi_gru_seq_fuse_pass.h"

#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/phi/core/errors.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using EigenVectorArrayMap = Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>>;
using string::PrettyLogDetail;

namespace {

std::vector<std::string> JoinInputs(Node* op1,
                                    Node* op2,
                                    std::string input_name) {
  auto in1 = op1->Op()->Input(input_name);
  auto& in2 = op2->Op()->Input(input_name);
  in1.insert(in1.end(), in2.begin(), in2.end());
  return in1;
}

}  // namespace

void MultiGruSeqFusePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Fusing two consecutive multi_gru ops.";
  PADDLE_ENFORCE_NOT_NULL(graph,
                          phi::errors::InvalidArgument(
                              "Pointer to graph argument cannot be NULL."));
  FusePassBase::Init(name_scope_, graph);
  PADDLE_ENFORCE_NOT_NULL(
      param_scope(), phi::errors::InvalidArgument("Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  patterns::MultiGruSeq pattern{gpd.mutable_pattern(), name_scope_};
  pattern();

  int fused_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(x, x, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gru1, gru1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wx11, wx11, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wx12, wx12, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wh11, wh11, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wh12, wh12, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(b11, b11, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(b12, b12, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(h1, h1, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gru2, gru2, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wx21, wx21, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wx22, wx22, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wh21, wh21, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wh22, wh22, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(b21, b21, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(b22, b22, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(h2, h2, pattern);

    if (gru1->Op()->GetAttrIfExists<bool>("origin_mode") !=
        gru2->Op()->GetAttrIfExists<bool>("origin_mode")) {
      LOG(INFO) << "The two multi_gru ops have different values of the "
                   "origin_mode attribute. Skipping fuse.";
      return;
    }

    auto wx = JoinInputs(gru1, gru2, "WeightX");
    auto wh = JoinInputs(gru1, gru2, "WeightH");
    auto b = JoinInputs(gru1, gru2, "Bias");

    OpDesc multi_gru_desc;
    multi_gru_desc.SetType("multi_gru");
    multi_gru_desc.SetInput("X", std::vector<std::string>({x->Name()}));
    multi_gru_desc.SetInput("WeightX", wx);
    multi_gru_desc.SetInput("WeightH", wh);
    multi_gru_desc.SetInput("Bias", b);
    multi_gru_desc.SetOutput("Hidden", std::vector<std::string>({h2->Name()}));

    for (auto& attr : gru1->Op()->GetAttrMap()) {
      multi_gru_desc.SetAttr(attr.first, attr.second);
    }

    auto layers = PADDLE_GET_CONST(int, gru1->Op()->GetAttr("layers")) +
                  PADDLE_GET_CONST(int, gru2->Op()->GetAttr("layers"));
    multi_gru_desc.SetAttr("layers", layers);

    auto multi_gru =
        g->CreateOpNode(&multi_gru_desc);  // OpDesc will be copied.

    IR_NODE_LINK_TO(x, multi_gru);
    IR_NODE_LINK_TO(wx11, multi_gru);
    IR_NODE_LINK_TO(wx12, multi_gru);
    IR_NODE_LINK_TO(wx21, multi_gru);
    IR_NODE_LINK_TO(wx22, multi_gru);
    IR_NODE_LINK_TO(wh11, multi_gru);
    IR_NODE_LINK_TO(wh12, multi_gru);
    IR_NODE_LINK_TO(wh21, multi_gru);
    IR_NODE_LINK_TO(wh22, multi_gru);
    IR_NODE_LINK_TO(b11, multi_gru);
    IR_NODE_LINK_TO(b12, multi_gru);
    IR_NODE_LINK_TO(b21, multi_gru);
    IR_NODE_LINK_TO(b22, multi_gru);
    IR_NODE_LINK_TO(multi_gru, h2);
    GraphSafeRemoveNodes(graph, {gru1, gru2, h1});

    ++fused_count;
  };
  gpd(graph, handler);
  AddStatis(fused_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs"))
    PrettyLogDetail("---    fused %d sequences of two multi_gru ops",
                    fused_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_gru_seq_fuse_pass,
              paddle::framework::ir::MultiGruSeqFusePass);
