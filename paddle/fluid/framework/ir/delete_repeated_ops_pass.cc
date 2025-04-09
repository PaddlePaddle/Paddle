// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace paddle::framework::ir {

bool HasOutVarName(Node* op_node, std::string name) {
  auto* op_desc = op_node->Op();
  auto outputs = op_desc->Outputs();
  for (auto const& iter : outputs) {
    auto out_names = iter.second;
    if (std::count(out_names.begin(), out_names.end(), name) > 0) {
      return true;
    }
  }
  return false;
}

}  // namespace paddle::framework::ir
namespace paddle::framework::ir::patterns {

struct VarWithRepeatedOpsPattern : public PatternBase {
  VarWithRepeatedOpsPattern(PDPattern* pattern,
                            const std::string& name_scope,
                            const std::string& op_type);

  // declare variable node's name
  PATTERN_DECL_NODE(in_var);

  std::string op_type_;
};

VarWithRepeatedOpsPattern::VarWithRepeatedOpsPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& op_type)
    : PatternBase(pattern, name_scope, name_scope), op_type_(op_type) {
  pattern->NewNode(in_var_repr())
      ->assert_is_var()
      ->assert_more([&](Node* node) {
        auto out_nodes = node->outputs;
        if (out_nodes.size() <= 1) return false;
        int op_counts = 0;
        for (auto* next_op : out_nodes) {
          if (next_op->Name() == op_type_) {
            op_counts++;
          }
        }
        return op_counts > 1;
      });
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

/*
Delete repeated ops, for example:
Origin subgraph:
     (input_variable)
      /     |    \     ...
    shape shape shape  ...
      |     |     |    ...
     op0   op1   op2   ...

Optimized subgraph:
      (input_variable)
            |
          shape
         /  |  \     ...
       op0 op1 op2   ...
*/
class DeleteRepeatedOpsPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void DeleteRepeatedOps(ir::Graph* graph,
                         const std::string& op_type,
                         std::function<std::string(Node*)> gen_op_key_fn) const;

  const std::string name_scope_{"delete_repeated_ops_pass"};
  mutable int delete_op_count{0};
};

void DeleteRepeatedOpsPass::DeleteRepeatedOps(
    ir::Graph* graph,
    const std::string& op_type,
    std::function<std::string(Node*)> gen_op_key_fn) const {
  GraphPatternDetector gpd;
  patterns::VarWithRepeatedOpsPattern pattern(
      gpd.mutable_pattern(), name_scope_, op_type);

  int delete_counts = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DeleteRepeatedOps";
    GET_IR_NODE_FROM_SUBGRAPH(in_var, in_var, pattern);
    // in_var node may be deleted by the previous detected subgraph
    if (graph->Nodes().count(in_var) == 0) {
      return;
    }

    std::vector<std::string> invalid_out_ops{
        "while", "conditional_block", "fetch"};
    std::map<std::string, std::vector<Node*>> ops_map;
    for (auto* next_op : in_var->outputs) {
      if (next_op->Name() != op_type) continue;
      auto* op = next_op;
      bool out_op_is_invalid = false;
      for (auto* out_op : op->outputs[0]->outputs) {
        if (std::count(invalid_out_ops.begin(),
                       invalid_out_ops.end(),
                       out_op->Name()) > 0 ||
            HasOutVarName(out_op, op->outputs[0]->Name())) {
          out_op_is_invalid = true;
          break;
        }
      }
      if (out_op_is_invalid) continue;
      auto attr_key = gen_op_key_fn(op);
      ops_map[attr_key].push_back(op);
    }
    for (auto iter = ops_map.begin(); iter != ops_map.end();) {
      if (iter->second.size() <= 1) {
        iter = ops_map.erase(iter);
      } else {
        iter++;
      }
    }

    for (auto const& iter : ops_map) {
      auto ops = iter.second;
      auto* first_op_out = ops[0]->outputs[0];
      auto first_op_out_name = first_op_out->Name();
      std::unordered_set<const Node*> delete_nodes;
      for (size_t i = 1; i < ops.size(); i++) {
        auto* cur_op = ops[i];
        auto* cur_op_out = cur_op->outputs[0];
        auto cur_op_out_name = cur_op_out->Name();
        for (auto* out_op : cur_op_out->outputs) {
          out_op->Op()->RenameInput(cur_op_out_name, first_op_out_name);
          IR_NODE_LINK_TO(first_op_out, out_op);
        }
        delete_nodes.insert(cur_op);
        delete_nodes.insert(cur_op_out);
        delete_counts++;
      }
      GraphSafeRemoveNodes(graph, delete_nodes);
    }
  };

  gpd(graph, handler);
  delete_op_count += delete_counts;
  if (delete_counts > 0) {
    LOG(INFO) << "--- delete " << delete_counts << " repeated " << op_type
              << " ops";
  }
}

std::string GenShapeAttrKey(Node* shape_op_node) { return ""; }

std::string GenSliceAttrKey(Node* slice_op_node) {
  std::string attr_key;
  auto slice_op_desc = slice_op_node->Op();
  auto starts = slice_op_desc->GetAttrIfExists<std::vector<int>>("starts");
  auto ends = slice_op_desc->GetAttrIfExists<std::vector<int>>("ends");
  auto axes = slice_op_desc->GetAttrIfExists<std::vector<int>>("axes");
  auto decrease_axis =
      slice_op_desc->GetAttrIfExists<std::vector<int>>("decrease_axis");
  attr_key += "starts_";
  for (auto start : starts) {
    attr_key += std::to_string(start) + "_";
  }
  attr_key += "ends_";
  for (auto end : ends) {
    attr_key += std::to_string(end) + "_";
  }
  attr_key += "axes_";
  for (auto axis : axes) {
    attr_key += std::to_string(axis) + "_";
  }
  attr_key += "decrease_axis_";
  for (auto axis : decrease_axis) {
    attr_key += std::to_string(axis) + "_";
  }
  return attr_key;
}

std::string GenCastAttrKey(Node* cast_op_node) {
  auto cast_op_desc = cast_op_node->Op();
  auto in_dtype = cast_op_desc->GetAttrIfExists<int>("in_dtype");
  auto out_dtype = cast_op_desc->GetAttrIfExists<int>("out_dtype");
  return "in_dtype_" + std::to_string(in_dtype) + "_out_dtype_" +
         std::to_string(out_dtype);
}

std::string GenAddAttrKey(Node* add_op_node) {
  auto add_op_desc = add_op_node->Op();
  std::string x_name = add_op_desc->Input("X")[0];
  std::string y_name = add_op_desc->Input("Y")[0];
  auto axis = add_op_desc->GetAttrIfExists<int>("axis");
  return x_name + "_" + y_name + "_axis_" + std::to_string(axis);
}

std::string GenTranspose2AttrKey(Node* transpose_op_node) {
  auto transpose_op_desc = transpose_op_node->Op();
  auto axis = transpose_op_desc->GetAttrIfExists<std::vector<int>>("axis");
  std::string attr_key;
  attr_key += "axis_";
  for (auto x : axis) {
    attr_key += std::to_string(x) + "_";
  }
  return attr_key;
}

std::string GenScaleAttrKey(Node* scale_op_node) {
  auto scale_op_desc = scale_op_node->Op();
  auto scale = scale_op_desc->GetAttrIfExists<float>("scale");
  auto bias = scale_op_desc->GetAttrIfExists<float>("bias");
  auto bias_after_scale =
      scale_op_desc->GetAttrIfExists<bool>("bias_after_scale");
  return "scale_" + std::to_string(scale) + "_bias_" + std::to_string(bias) +
         "_bias_after_scale_" + std::to_string(bias_after_scale);
}

std::string GenGatherAttrKey(Node* gather_op_node) {
  std::string input_names{""};
  for (auto input_var : gather_op_node->inputs) {
    input_names += input_var->Var()->Name();
  }
  auto gather_op_desc = gather_op_node->Op();
  auto axis = gather_op_desc->GetAttrIfExists<int>("axis");
  return "axis_" + std::to_string(axis) + "_input_names_" + input_names;
}

std::string GenSqueeze2AttrKey(Node* squeeze2_op_node) {
  auto squeeze2_op_desc = squeeze2_op_node->Op();
  auto axes = squeeze2_op_desc->GetAttrIfExists<std::vector<int>>("axes");
  std::string attr_key{""};
  attr_key += "axes_";
  for (auto axis : axes) {
    attr_key += std::to_string(axis) + "_";
  }
  return attr_key;
}

void DeleteRepeatedOpsPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  int repeat_time = 0;
  int total_delete_op_count = 0;
  // This pass needs to loop run until there are no nodes in the graph that need
  // to be deleted.
  while (true) {
    delete_op_count = 0;
    DeleteRepeatedOps(graph, "shape", GenShapeAttrKey);
    DeleteRepeatedOps(graph, "slice", GenSliceAttrKey);
    DeleteRepeatedOps(graph, "cast", GenCastAttrKey);
    DeleteRepeatedOps(graph, "elementwise_add", GenAddAttrKey);
    DeleteRepeatedOps(graph, "scale", GenScaleAttrKey);
    DeleteRepeatedOps(graph, "gather", GenGatherAttrKey);
    DeleteRepeatedOps(graph, "squeeze2", GenSqueeze2AttrKey);
    DeleteRepeatedOps(graph, "unsqueeze2", GenSqueeze2AttrKey);
    DeleteRepeatedOps(graph, "transpose2", GenTranspose2AttrKey);
    LOG(INFO) << "Round " << repeat_time++
              << ": delete op counts: " << delete_op_count;
    total_delete_op_count += delete_op_count;
    if (delete_op_count == 0) {
      break;  // No node need to delete.
    }
  }
  LOG(INFO) << "Total delete op counts: " << total_delete_op_count;
}

}  // namespace paddle::framework::ir

REGISTER_PASS(delete_repeated_ops_pass,
              paddle::framework::ir::DeleteRepeatedOpsPass);

REGISTER_PASS_CAPABILITY(delete_repeated_ops_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "shape", 0));
