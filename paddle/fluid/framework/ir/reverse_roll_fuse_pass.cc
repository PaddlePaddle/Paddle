// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/reverse_roll_fuse_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, reverse_roll_pattern);
#define GET_NODES   \
  GET_IR_NODE(window_mha_i00_op);       \
  GET_IR_NODE(window_mha_i00_out);      \
  GET_IR_NODE(matmul_i10_op);           \
  GET_IR_NODE(matmul_i10_out);          \
  GET_IR_NODE(elw_add_i20_op);          \
  GET_IR_NODE(elw_add_i20_out);         \
  GET_IR_NODE(reshape2_00_op);          \
  GET_IR_NODE(reshape2_00_out);         \
  GET_IR_NODE(reshape2_10_op);          \
  GET_IR_NODE(reshape2_10_out);         \
  GET_IR_NODE(transpose2_20_op);        \
  GET_IR_NODE(transpose2_20_out);       \
  GET_IR_NODE(reshape2_30_op);          \
  GET_IR_NODE(reshape2_30_out);         \
  GET_IR_NODE(roll_40_op);              \
  GET_IR_NODE(roll_40_out);             \
  GET_IR_NODE(reshape2_50_op);          \
  GET_IR_NODE(reshaep2_50_out);         


namespace paddle {
namespace framework {
namespace ir {
    void ReverseRollFusePass::ApplyImpl(ir::Graph* graph) const {
        GraphPatternDetector gpd;
        const std::string pattern_name = "reverse_roll_fuse";
        FusePassBase::Init(pattern_name, graph);
        // auto* scope = param_scope();
        PDNode* x = gpd.mutable_pattern()
                    ->NewNode("x")
                    ->assert_is_op_input("multihead_matmul", "X")
                    ->AsInput();
        patterns::ReverseRollPattern reverse_roll_pattern(
            gpd.mutable_pattern(), scope_name_);
        reverse_roll_pattern(x);
        int fuse_count = 0;
        auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
            GET_NODES;
            OpDesc reverse_roll_desc(reshape2_00_op->Op()->Block());
            reverse_roll_desc.SetType("reverse_roll");
            reverse_roll_desc.SetInput("X",{elw_add_i20_out->Name()});
            reverse_roll_desc.SetOutput("Out",{reshaep2_50_out->Name()});
            auto reverse_roll_node = graph->CreateOpNode(&reverse_roll_desc);
            IR_NODE_LINK_TO(elw_add_i20_out,reverse_roll_node);
            IR_NODE_LINK_TO(reverse_roll_node,reshaep2_50_out);
            GraphSafeRemoveNodes(graph,{
                reshape2_00_op,   
                reshape2_00_out,  
                reshape2_10_op,   
                reshape2_10_out,  
                transpose2_20_op, 
                transpose2_20_out,
                reshape2_30_op,   
                reshape2_30_out,  
                roll_40_op,       
                roll_40_out,      
                reshape2_50_op});
            ++fuse_count;
        };
        gpd(graph, handler);
        AddStatis(fuse_count);     
    }
} // ir
} // framework
} // paddle 

REGISTER_PASS(reverse_roll_fuse_pass,
              paddle::framework::ir::ReverseRollFusePass);
REGISTER_PASS_CAPABILITY(reverse_roll_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("transpose2", 0)
            .EQ("reshape2", 0));