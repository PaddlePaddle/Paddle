/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/framework/ir/seqpool_cvm_concat_fuse_pass.h"
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"

#define MAX_CONCAT_INPUTS 500

namespace paddle {
namespace framework {
namespace ir {

PDNode* BuildSeqPoolCVMConcatPattern(PDPattern* pattern,
                                     const std::string& name_scope,
                                     int num_inputs) {
  auto is_concat_op_with_inputs = [](Node* x, int num) -> bool {
    return x && x->IsOp() && x->Op()->Type() == "concat" &&
           x->Op()->Input("X").size() == static_cast<size_t>(num);
  };

  auto is_nth_input_var_of_concat = [=](Node* x, int idx) -> bool {
    return x && x->IsVar() && VarLinksToOp(x, "concat") &&
           x->outputs.size() == 1 && IsNthInput(x, x->outputs[0], "X", idx) &&
           is_concat_op_with_inputs(x->outputs[0], num_inputs);
  };

  auto is_input_var_of_cvm = [=](Node* x, std::string arg) -> bool {
    return x && x->IsVar() && VarLinksToOp(x, "cvm") &&
           x->outputs.size() == 1 && IsNthInput(x, x->outputs[0], arg, 0);
  };

  auto is_cvm_op = [=](Node* x, bool use_cvm) -> bool {
    return x && x->IsOp() && x->Op()->Type() == "cvm" &&
           x->Op()->HasAttr("use_cvm") &&
           boost::get<bool>(x->Op()->GetAttr("use_cvm")) == use_cvm &&
           x->Op()->Input("X").size() == 1 &&
           x->Op()->Input("CVM").size() == 1 &&
           x->Op()->Output("Y").size() == 1;
  };

  auto is_cvm_op_on_nth_branch_input_of_concat = [=](Node* x, bool use_cvm,
                                                     int idx) -> bool {
    return is_cvm_op(x, use_cvm) &&
           is_nth_input_var_of_concat(x->outputs[0], idx);
  };

  auto is_x_input_var_of_cvm_op_on_nth_branch = [=](Node* x, bool use_cvm,
                                                    int idx) -> bool {
    return is_input_var_of_cvm(x, "X") &&
           is_cvm_op_on_nth_branch_input_of_concat(x->outputs[0], use_cvm, idx);
  };

  auto is_seqpool_op_on_nth_branch_before_cvm = [=](
      Node* x, const std::string& pooltype, bool use_cvm, int idx) -> bool {
    bool this_is_seqpool_op =
        x && x->IsOp() && x->Op()->Type() == "sequence_pool" &&
        x->Op()->HasAttr("pooltype") &&
        boost::get<std::string>(x->Op()->GetAttr("pooltype")) == pooltype &&
        x->outputs.size() == 2;  // seqpool should only have 2 outputs
    bool satisfied_all = this_is_seqpool_op;
    if (this_is_seqpool_op) {
      if (is_x_input_var_of_cvm_op_on_nth_branch(x->outputs[0], use_cvm, idx)) {
        satisfied_all = satisfied_all && x->outputs[1]->IsVar() &&
                        x->outputs[1]->outputs.empty();
      } else {
        satisfied_all = satisfied_all && is_x_input_var_of_cvm_op_on_nth_branch(
                                             x->outputs[1], use_cvm, idx) &&
                        x->outputs[0]->IsVar() &&
                        x->outputs[0]->outputs.size() == 0;
      }
    }
    return satisfied_all;
  };

  auto* concat_op = pattern->NewNode(
      [=](Node* x) { return is_concat_op_with_inputs(x, num_inputs); },
      name_scope + "/concat_op");
  concat_op->assert_op_attr<int>("axis", 1);

  auto* concat_out_var = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsVar() && VarLinksFromOp(x, "concat") &&
               x->inputs.size() == 1 &&
               is_concat_op_with_inputs(x->inputs[0], num_inputs);
      },
      name_scope + "/concat_out_var");
  concat_out_var->assert_is_only_output_of_op("concat");

  std::vector<PDNode*> seqpool_ops_input_var(num_inputs);
  std::vector<PDNode*> cvm_ops_input_var(num_inputs);
  std::vector<PDNode*> seqpool_ops_output_unused_var(num_inputs);
  std::vector<PDNode*> seqpool_ops(num_inputs);
  std::vector<PDNode*> cvm_ops(num_inputs);
  std::vector<PDNode*> cvm_ops_output_var(num_inputs);
  PDNode* cvm_ops_input_cvm_var;

  cvm_ops_input_cvm_var = pattern->NewNode(
      [=](Node* x) {
        return x && x->IsVar() &&
               x->outputs.size() == static_cast<size_t>(num_inputs) &&
               is_cvm_op(x->outputs[0], true);
      },
      name_scope + "/cvm_ops_input_cvm_var");

  for (int i = 0; i < num_inputs; ++i) {
    cvm_ops_output_var[i] = pattern->NewNode(
        [=](Node* x) {
          return x && x->IsVar() && x->inputs.size() == 1 &&
                 is_cvm_op_on_nth_branch_input_of_concat(x->inputs[0], true, i);
        },
        name_scope + "/cvm_ops_output_" + std::to_string(i));

    cvm_ops[i] = pattern->NewNode(
        [=](Node* x) {
          return x && x->IsOp() &&
                 is_cvm_op_on_nth_branch_input_of_concat(x, true, i);
        },
        name_scope + "/cvm_op_" + std::to_string(i));

    cvm_ops_input_var[i] = pattern->NewNode(
        [=](Node* x) {
          return x && x->IsVar() && x->inputs.size() == 1 &&
                 is_seqpool_op_on_nth_branch_before_cvm(x->inputs[0], "SUM",
                                                        true, i);
        },
        name_scope + "/cvm_ops_input_" + std::to_string(i));

    seqpool_ops_output_unused_var[i] = pattern->NewNode(
        [=](Node* x) {
          return x && x->IsVar() && x->inputs.size() == 1 &&
                 x->outputs.size() == 0 &&
                 is_seqpool_op_on_nth_branch_before_cvm(x->inputs[0], "SUM",
                                                        true, i);
        },
        name_scope + "/sequence_pool_unused_out_" + std::to_string(i));

    seqpool_ops[i] = pattern->NewNode(
        [=](Node* x) {
          return x && x->IsOp() &&
                 is_seqpool_op_on_nth_branch_before_cvm(x, "SUM", true, i);
        },
        name_scope + "/sequence_pool_op_" + std::to_string(i));

    seqpool_ops_input_var[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() && x->outputs.size() >= 1;
          bool next_is_fine = false;
          for (auto* o : x->outputs) {
            if (is_seqpool_op_on_nth_branch_before_cvm(o, "SUM", true, i)) {
              next_is_fine = true;
              break;
            }
          }
          return basic && next_is_fine;
        },
        name_scope + "/sequence_pool_in_" + std::to_string(i));

    // Links
    seqpool_ops[i]
        ->LinksFrom({seqpool_ops_input_var[i]})
        .LinksTo({cvm_ops_input_var[i], seqpool_ops_output_unused_var[i]});

    cvm_ops[i]
        ->LinksFrom({cvm_ops_input_var[i], cvm_ops_input_cvm_var})
        .LinksTo({cvm_ops_output_var[i]});
  }
  concat_op->LinksFrom(cvm_ops_output_var).LinksTo({concat_out_var});
  return concat_out_var;
}

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       int num_inputs) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  BuildSeqPoolCVMConcatPattern(pattern, name_scope, num_inputs);

  auto retrieve_node = [](const std::string& name,
                          const GraphPatternDetector::subgraph_t& subgraph,
                          const PDPattern& pat) -> Node* {
    PADDLE_ENFORCE(subgraph.count(pat.RetrieveNode(name)),
                   "pattern has no Node called %s", name.c_str());
    Node* p = subgraph.at(pat.RetrieveNode(name));
    PADDLE_ENFORCE_NOT_NULL(p, "subgraph has no node %s", name.c_str());
    return p;
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle SeqPool CVM Concat fuse";
    std::vector<std::string> input_names(num_inputs);
    std::vector<Node*> input_vars(num_inputs);
    auto& fused_pattern = gpd.pattern();
    for (int i = 0; i < num_inputs; ++i) {
      input_vars[i] =
          retrieve_node(name_scope + "/sequence_pool_in_" + std::to_string(i),
                        subgraph, fused_pattern);
      input_names[i] = input_vars[i]->Name();
    }
    auto* concat_op =
        retrieve_node(name_scope + "/concat_op", subgraph, fused_pattern);
    auto* concat_out_var =
        retrieve_node(name_scope + "/concat_out_var", subgraph, fused_pattern);
    auto* seqpool_op0 = retrieve_node(name_scope + "/sequence_pool_op_0",
                                      subgraph, fused_pattern);
    auto* cvm_op0 =
        retrieve_node(name_scope + "/cvm_op_0", subgraph, fused_pattern);
    auto* cvm_ops_input_cvm_var = retrieve_node(
        name_scope + "/cvm_ops_input_cvm_var", subgraph, fused_pattern);

    // Create New OpDesc
    OpDesc op_desc;
    op_desc.SetType("fusion_seqpool_cvm_concat");
    op_desc.SetInput("X", input_names);
    op_desc.SetInput("CVM", {cvm_ops_input_cvm_var->Name()});
    op_desc.SetAttr("pooltype", seqpool_op0->Op()->GetAttr("pooltype"));
    op_desc.SetAttr("use_cvm", cvm_op0->Op()->GetAttr("use_cvm"));
    op_desc.SetAttr("axis", concat_op->Op()->GetAttr("axis"));
    op_desc.SetOutput("Out", {concat_out_var->Name()});
    auto* op = graph->CreateOpNode(&op_desc);
    for (size_t i = 0; i < input_vars.size(); ++i) {
      IR_NODE_LINK_TO(input_vars[i], op);
    }
    IR_NODE_LINK_TO(cvm_ops_input_cvm_var, op);
    IR_NODE_LINK_TO(op, concat_out_var);

    std::unordered_set<const Node*> marked_nodes;
    for (auto& item : subgraph) {
      marked_nodes.insert(item.second);
    }
    for (size_t i = 0; i < input_vars.size(); ++i) {
      marked_nodes.erase(input_vars[i]);
    }
    marked_nodes.erase(concat_out_var);
    marked_nodes.erase(cvm_ops_input_cvm_var);
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };

  gpd(graph, handler);
  return fusion_count;
}

void SeqPoolCVMConcatFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  int fusion_count = 0;
  for (int i = MAX_CONCAT_INPUTS; i > 0; --i) {
    fusion_count +=
        BuildFusion(graph, name_scope_ + "/" + std::to_string(i), i);
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(seqpool_cvm_concat_fuse_pass,
              paddle::framework::ir::SeqPoolCVMConcatFusePass);
