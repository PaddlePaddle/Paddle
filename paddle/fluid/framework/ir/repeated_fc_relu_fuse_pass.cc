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

#include "paddle/fluid/framework/ir/repeated_fc_relu_fuse_pass.h"
#include <algorithm>  // for max
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"

#define MAX_NUM_FC 10

namespace paddle {
namespace framework {
namespace ir {

PDNode* BuildRepeatedFCReluPattern(PDPattern* pattern,
                                   const std::string& name_scope, int num_fc) {
  auto var_next_is_fc_act = [=](Node* x, const std::string& act_type = "relu",
                                bool check_in_has_only_one_out = true,
                                int fc_idx = 0) -> bool {
    bool next_is_fc = x && x->IsVar() && VarLinksToOp(x, "fc");
    if (check_in_has_only_one_out) {
      next_is_fc = next_is_fc && x->outputs.size() == 1;
    }
    if (!next_is_fc) {
      return false;
    }
    auto* fc_op = x->outputs[fc_idx];
    bool next_is_act = fc_op && fc_op->IsOp() && fc_op->outputs.size() == 1 &&
                       fc_op->outputs[0] && fc_op->outputs[0]->IsVar() &&
                       VarLinksToOp(fc_op->outputs[0], act_type) &&
                       fc_op->outputs[0]->outputs.size() == 1;
    if (!next_is_act) {
      return false;
    }
    auto* act_op = fc_op->outputs[0]->outputs[0];
    return act_op && act_op->IsOp() && act_op->outputs.size() == 1;
  };

  auto find_fc_idx = [=](Node* x, const std::string& act_type = "relu") -> int {
    bool next_is_fc = x && x->IsVar() && VarLinksToOp(x, "fc");
    if (!next_is_fc) {
      return 0;
    }
    for (size_t k = 0; k < x->outputs.size(); ++k) {
      auto* fc_op = x->outputs[k];
      bool next_is_act = fc_op && fc_op->IsOp() && fc_op->outputs.size() == 1 &&
                         fc_op->outputs[0] && fc_op->outputs[0]->IsVar() &&
                         VarLinksToOp(fc_op->outputs[0], act_type) &&
                         fc_op->outputs[0]->outputs.size() == 1;
      if (!next_is_act) {
        continue;
      }
      auto* act_op = fc_op->outputs[0]->outputs[0];
      if (act_op && act_op->IsOp() && act_op->outputs.size() == 1) {
        return k;
      }
    }
    return 0;
  };

  auto next_var_of_part = [=](Node* x, int fc_idx = 0) -> Node* {
    return x->outputs[fc_idx]->outputs[0]->outputs[0]->outputs[0];
  };
  auto var_next_is_fc_act_repeated_n_times = [=](
      Node* x, int repeated_times, const std::string& act_type = "relu",
      bool check_in_has_only_one_out = true) -> bool {
    for (int i = 0; i < repeated_times; ++i) {
      if (!var_next_is_fc_act(x, act_type,
                              i == 0 && check_in_has_only_one_out)) {
        return false;
      }
      x = next_var_of_part(x);
    }
    return true;
  };

  auto var_before_is_fc_act = [=](Node* x, const std::string& act_type = "relu",
                                  bool at_top = false) -> bool {
    bool before_is_act =
        x && x->IsVar() && x->inputs.size() == 1 && VarLinksFromOp(x, "relu");
    if (!before_is_act) {
      return false;
    }
    auto* relu_op = x->inputs[0];
    bool before_is_fc = relu_op->IsOp() && relu_op->inputs.size() == 1 &&
                        relu_op->inputs[0]->IsVar() &&
                        VarLinksFromOp(relu_op->inputs[0], "fc") &&
                        relu_op->inputs[0]->inputs.size() == 1;

    if (!before_is_fc) {
      return false;
    }
    auto* fc_op = relu_op->inputs[0]->inputs[0];
    bool is_fc = fc_op->IsOp() && fc_op->inputs.size() == 3;
    if (!is_fc) {
      return false;
    }
    for (auto* fc_i : fc_op->inputs) {
      if (!fc_i->inputs.empty()) {
        if (at_top) {
          return true;
        } else {
          return VarLinksFromOp(fc_i, "relu");
        }
      }
    }
    return false;
  };

  auto before_var_of_part = [=](Node* x) -> Node* {
    auto* fc_op = x->inputs[0]->inputs[0];
    for (auto* fc_i : fc_op->inputs) {
      if (!fc_i->inputs.empty()) {
        return fc_i->inputs[0];
      }
    }
    return nullptr;
  };

  auto var_before_is_fc_act_repeated_n_times = [=](
      Node* x, int repeated_times,
      const std::string& act_type = "relu") -> bool {
    for (int i = 0; i < repeated_times; ++i) {
      if (!var_before_is_fc_act(x, act_type, i == repeated_times - 1)) {
        return false;
      }
      x = before_var_of_part(x);
    }
    return true;
  };

  std::vector<PDNode*> fc_input_var(num_fc);
  std::vector<PDNode*> fc_output_var(num_fc);
  std::vector<PDNode*> fc_weight_var(num_fc);
  std::vector<PDNode*> fc_bias_var(num_fc);
  std::vector<PDNode*> fc_ops(num_fc);
  std::vector<PDNode*> relu_ops(num_fc);

  for (int i = 0; i < num_fc; ++i) {
    fc_input_var[i] = pattern->NewNode(
        [=](Node* x) {
          if (i == 0 && x->outputs.size() > 0) {
            bool ok = x->inputs.size() > 0;
            if (!ok) {
              return false;
            }
            int idx = find_fc_idx(x);
            if (idx == 0) {
              return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu");
            } else {
              x = next_var_of_part(x, idx);
              return var_next_is_fc_act_repeated_n_times(
                  x, std::max(1, num_fc - i - 1), "relu");
            }
          } else {
            return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu") &&
                   x->inputs.size() > 0 &&
                   var_before_is_fc_act_repeated_n_times(x, i, "relu");
          }
        },
        name_scope + "/fc_in_" + std::to_string(i));

    fc_weight_var[i] = pattern->NewNode(
        [=](Node* x) {
          return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu") &&
                 x->inputs.empty() &&
                 var_before_is_fc_act_repeated_n_times(x->outputs[0]->inputs[0],
                                                       i, "relu") &&
                 x->Name() == x->outputs[0]->Op()->Input("W")[0];
        },
        name_scope + "/fc_weight_" + std::to_string(i));

    fc_bias_var[i] = pattern->NewNode(
        [=](Node* x) {
          return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu") &&
                 x->inputs.empty() &&
                 var_before_is_fc_act_repeated_n_times(x->outputs[0]->inputs[0],
                                                       i, "relu") &&
                 x->Name() == x->outputs[0]->Op()->Input("Bias")[0];
        },
        name_scope + "/fc_bias_" + std::to_string(i));

    fc_output_var[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsVar() && VarLinksFromOp(x, "fc") &&
                       VarLinksToOp(x, "relu") && x->inputs.size() == 1 &&
                       x->inputs[0]->inputs.size() == 3;
          if (!basic) {
            return false;
          }
          x = x->inputs[0]->inputs[0];
          if (i == 0 && x->outputs.size() > 0) {
            bool ok = x->inputs.size() > 0;
            if (!ok) {
              return false;
            }
            int idx = find_fc_idx(x);
            if (idx == 0) {
              return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu");
            } else {
              x = next_var_of_part(x, idx);
              return var_next_is_fc_act_repeated_n_times(
                  x, std::max(1, num_fc - i - 1), "relu");
            }
          } else {
            return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu") &&
                   x->inputs.size() > 0 &&
                   var_before_is_fc_act_repeated_n_times(x, i, "relu");
          }
        },
        name_scope + "/fc_out_" + std::to_string(i));

    fc_ops[i] = pattern->NewNode(
        [=](Node* x) {
          bool basic = x && x->IsOp() && x->Op()->Type() == "fc" &&
                       x->inputs.size() == 3 && x->outputs.size() == 1;
          if (!basic) {
            return false;
          }
          auto* fc_out_var = x->outputs[0];
          return fc_out_var && fc_out_var->IsVar() &&
                 fc_out_var->outputs.size() == 1 &&
                 VarLinksToOp(fc_out_var, "relu") &&
                 fc_out_var->outputs[0]->outputs.size() == 1 &&
                 var_next_is_fc_act_repeated_n_times(
                     fc_out_var->outputs[0]->outputs[0], num_fc - i - 1,
                     "relu") &&
                 var_before_is_fc_act_repeated_n_times(
                     fc_out_var->outputs[0]->outputs[0], i + 1, "relu");
        },
        name_scope + "/fc_op_" + std::to_string(i));

    relu_ops[i] = pattern->NewNode(
        [=](Node* x) {
          return x && x->IsOp() && x->Op()->Type() == "relu" &&
                 x->inputs.size() == 1 && x->outputs.size() == 1 &&
                 x->inputs[0]->IsVar() && VarLinksFromOp(x->inputs[0], "fc") &&
                 x->outputs[0]->IsVar() &&
                 var_next_is_fc_act_repeated_n_times(x->outputs[0],
                                                     num_fc - i - 1, "relu") &&
                 var_before_is_fc_act_repeated_n_times(x->outputs[0], i + 1,
                                                       "relu");
        },
        name_scope + "/act_op_" + std::to_string(i));

    fc_ops[i]
        ->LinksFrom({fc_input_var[i], fc_weight_var[i], fc_bias_var[i]})
        .LinksTo({fc_output_var[i]});
    relu_ops[i]->LinksFrom({fc_output_var[i]});
  }

  auto* last_out_var = pattern->NewNode(
      [=](Node* x) {
        return var_before_is_fc_act_repeated_n_times(x, num_fc, "relu");
      },
      name_scope + "/act_out");
  for (int i = 0; i < num_fc - 1; ++i) {
    relu_ops[i]->LinksTo({fc_input_var[i + 1]});
  }
  relu_ops[num_fc - 1]->LinksTo({last_out_var});
  return last_out_var;
}

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       int num_fc) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  BuildRepeatedFCReluPattern(pattern, name_scope, num_fc);

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
    LOG(INFO) << "handle Repeated FC Act fuse";
    std::vector<Node*> weights_vars(num_fc);
    std::vector<Node*> bias_vars(num_fc);
    std::vector<Node*> relu_vars(num_fc - 1);

    std::vector<std::string> weight_names(num_fc);
    std::vector<std::string> bias_names(num_fc);
    std::vector<std::string> relu_names(num_fc - 1);

    auto& fused_pattern = gpd.pattern();
    for (int i = 0; i < num_fc; ++i) {
      if (i >= 1) {
        relu_vars[i - 1] =
            retrieve_node(name_scope + "/fc_in_" + std::to_string(i), subgraph,
                          fused_pattern);
        relu_names[i - 1] = relu_vars[i - 1]->Name();
      }

      weights_vars[i] =
          retrieve_node(name_scope + "/fc_weight_" + std::to_string(i),
                        subgraph, fused_pattern);
      weight_names[i] = weights_vars[i]->Name();

      bias_vars[i] = retrieve_node(name_scope + "/fc_bias_" + std::to_string(i),
                                   subgraph, fused_pattern);
      bias_names[i] = bias_vars[i]->Name();
    }

    auto* input_var =
        retrieve_node(name_scope + "/fc_in_0", subgraph, fused_pattern);
    auto* last_out_var =
        retrieve_node(name_scope + "/act_out", subgraph, fused_pattern);

    // Create New OpDesc
    OpDesc op_desc;
    op_desc.SetType("fusion_repeated_fc_relu");
    op_desc.SetInput("X", {input_var->Name()});
    op_desc.SetInput("W", weight_names);
    op_desc.SetInput("Bias", bias_names);
    op_desc.SetOutput("ReluOut", relu_names);
    op_desc.SetOutput("Out", {last_out_var->Name()});
    auto* op = graph->CreateOpNode(&op_desc);
    IR_NODE_LINK_TO(input_var, op);
    for (size_t i = 0; i < weights_vars.size(); ++i) {
      IR_NODE_LINK_TO(weights_vars[i], op);
      IR_NODE_LINK_TO(bias_vars[i], op);
    }
    for (size_t i = 0; i < relu_vars.size(); ++i) {
      IR_NODE_LINK_TO(op, relu_vars[i]);
    }
    IR_NODE_LINK_TO(op, last_out_var);

    std::unordered_set<const Node*> marked_nodes;
    for (auto& item : subgraph) {
      marked_nodes.insert(item.second);
    }
    for (size_t i = 0; i < weights_vars.size(); ++i) {
      marked_nodes.erase(weights_vars[i]);
      marked_nodes.erase(bias_vars[i]);
    }
    for (size_t i = 0; i < relu_vars.size(); ++i) {
      marked_nodes.erase(relu_vars[i]);
    }
    marked_nodes.erase(input_var);
    marked_nodes.erase(last_out_var);
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };

  gpd(graph, handler);
  return fusion_count;
}

ir::Graph* RepeatedFCReluFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  int fusion_count = 0;
  for (int i = MAX_NUM_FC; i > 1; --i) {
    fusion_count +=
        BuildFusion(graph, name_scope_ + "/" + std::to_string(i), i);
  }
  AddStatis(fusion_count);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(repeated_fc_relu_fuse_pass,
              paddle::framework::ir::RepeatedFCReluFusePass);
