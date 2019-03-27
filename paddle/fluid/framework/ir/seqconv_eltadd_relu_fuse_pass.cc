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

#include "paddle/fluid/framework/ir/seqconv_eltadd_relu_fuse_pass.h"
#include <string>
#include <unordered_set>
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {
namespace ir {

int BuildFusion(Graph* graph, const std::string& name_scope, Scope* scope) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  PDNode* x = pattern->NewNode(patterns::PDNodeName(name_scope, "X"))
                  ->assert_is_op_input("sequence_conv")
                  ->assert_var_not_persistable();
  patterns::SeqConvEltAddRelu fuse_pattern(pattern, name_scope);
  fuse_pattern(x);

  // Create New OpDesc
  auto fuse_creator = [&](Node* seqconv, Node* input, Node* seqconv_weight,
                          Node* eltadd_bias, Node* relu_out) {
    OpDesc op_desc;
    op_desc.SetType("fusion_seqconv_eltadd_relu");
    op_desc.SetInput("X", {input->Name()});
    op_desc.SetInput("Filter", {seqconv_weight->Name()});
    op_desc.SetInput("Bias", {eltadd_bias->Name()});
    op_desc.SetAttr("contextLength", seqconv->Op()->GetAttr("contextLength"));
    op_desc.SetAttr("contextStart", seqconv->Op()->GetAttr("contextStart"));
    op_desc.SetAttr("contextStride", seqconv->Op()->GetAttr("contextStride"));
    PADDLE_ENFORCE(graph->Has(kParamScopeAttr));
    auto* scope = graph->Get<Scope*>(kParamScopeAttr);
    const std::string ColMat = patterns::UniqueKey("SeqConvColMat");
    op_desc.SetOutput("ColMat", {ColMat});
    op_desc.SetOutput("Out", {relu_out->Name()});
    scope->Var(ColMat)->GetMutable<LoDTensor>();

    auto* op = graph->CreateOpNode(&op_desc);
    IR_NODE_LINK_TO(input, op);
    IR_NODE_LINK_TO(seqconv_weight, op);
    IR_NODE_LINK_TO(eltadd_bias, op);
    IR_NODE_LINK_TO(op, relu_out);
    return op;
  };

  int fusion_count{0};

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle SeqConv EltAdd Relu fuse";
    GET_IR_NODE_FROM_SUBGRAPH(seqconv, seqconv, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(seqconv_weight, seqconv_weight, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(seqconv_out, seqconv_out, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd, eltadd, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_bias, eltadd_bias, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_out, eltadd_out, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(relu, relu, fuse_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(relu_out, relu_out, fuse_pattern);

    fuse_creator(seqconv, subgraph.at(x), seqconv_weight, eltadd_bias,
                 relu_out);
    std::unordered_set<const Node*> marked_nodes(
        {seqconv, seqconv_out, eltadd, eltadd_out, relu});
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };

  gpd(graph, handler);

  return fusion_count;
}

ir::Graph* SeqConvEltAddReluFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count = BuildFusion(graph, name_scope_, param_scope());
  AddStatis(fusion_count);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(seqconv_eltadd_relu_fuse_pass,
              paddle::framework::ir::SeqConvEltAddReluFusePass);
