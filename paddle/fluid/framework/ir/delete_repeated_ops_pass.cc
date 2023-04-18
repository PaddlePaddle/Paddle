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

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

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

}  // namespace patterns

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
  int DeleteShapePass(ir::Graph* graph) const;

  int DeleteSlicePass(ir::Graph* graph) const;

  const std::string name_scope_{"delete_repeated_ops_pass"};
};

int DeleteRepeatedOpsPass::DeleteShapePass(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::VarWithRepeatedOpsPattern pattern(
      gpd.mutable_pattern(), name_scope_, "shape");

  int delete_counts = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DeleteShapePass";
    GET_IR_NODE_FROM_SUBGRAPH(in_var, in_var, pattern);

    std::vector<Node*> shapes;
    for (auto* next_op : in_var->outputs) {
      if (next_op->Name() != "shape") continue;
      bool shape_out_has_control_flow_ops = false;
      for (auto* shape_out_op : next_op->outputs[0]->outputs) {
        if (shape_out_op->Name() == "while" ||
            shape_out_op->Name() == "conditional_block") {
          shape_out_has_control_flow_ops = true;
          break;
        }
      }
      if (!shape_out_has_control_flow_ops) {
        shapes.push_back(next_op);
      }
    }
    if (shapes.size() <= 1) return;

    auto* first_shape_out = shapes[0]->outputs[0];
    auto first_shape_out_name = first_shape_out->Name();
    std::unordered_set<const Node*> delete_nodes;
    for (size_t i = 1; i < shapes.size(); i++) {
      auto* cur_shape = shapes[i];
      auto* cur_shape_out = cur_shape->outputs[0];
      auto cur_shape_out_name = cur_shape_out->Name();
      for (auto* shape_out_op : cur_shape_out->outputs) {
        shape_out_op->Op()->Rename(cur_shape_out_name, first_shape_out_name);
        IR_NODE_LINK_TO(first_shape_out, shape_out_op);
      }
      delete_nodes.insert(cur_shape);
      delete_nodes.insert(cur_shape_out);
      delete_counts++;
    }

    GraphSafeRemoveNodes(graph, delete_nodes);
  };

  gpd(graph, handler);
  return delete_counts;
}

std::string GenSliceAttrKey(OpDesc* slice_op_desc) {
  std::string attr_key;
  auto starts = slice_op_desc->GetAttrIfExists<std::vector<int>>("starts");
  auto ends = slice_op_desc->GetAttrIfExists<std::vector<int>>("ends");
  auto axes = slice_op_desc->GetAttrIfExists<std::vector<int>>("axes");
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
  return attr_key;
}

int DeleteRepeatedOpsPass::DeleteSlicePass(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::VarWithRepeatedOpsPattern pattern(
      gpd.mutable_pattern(), name_scope_, "slice");

  int delete_counts = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DeleteSlicePass";
    GET_IR_NODE_FROM_SUBGRAPH(in_var, in_var, pattern);

    std::map<std::string, std::vector<Node*>> slice_ops;
    for (auto* next_op : in_var->outputs) {
      if (next_op->Name() != "slice") continue;
      auto* slice = next_op;
      bool slice_out_has_control_flow_ops = false;
      for (auto* slice_out_op : slice->outputs[0]->outputs) {
        if (slice_out_op->Name() == "while" ||
            slice_out_op->Name() == "conditional_block") {
          slice_out_has_control_flow_ops = true;
          break;
        }
      }
      if (slice_out_has_control_flow_ops) continue;
      auto attr_key = GenSliceAttrKey(slice->Op());
      slice_ops[attr_key].push_back(slice);
    }
    for (auto iter = slice_ops.begin(); iter != slice_ops.end();) {
      if (iter->second.size() <= 1) {
        iter = slice_ops.erase(iter);
      } else {
        iter++;
      }
    }

    for (auto iter : slice_ops) {
      auto slices = iter.second;
      auto* first_slice_out = slices[0]->outputs[0];
      auto first_slice_out_name = first_slice_out->Name();
      std::unordered_set<const Node*> delete_nodes;
      for (size_t i = 1; i < slices.size(); i++) {
        auto* cur_slice = slices[i];
        auto* cur_slice_out = cur_slice->outputs[0];
        auto cur_slice_out_name = cur_slice_out->Name();
        for (auto* slice_out_op : cur_slice_out->outputs) {
          slice_out_op->Op()->Rename(cur_slice_out_name, first_slice_out_name);
          IR_NODE_LINK_TO(first_slice_out, slice_out_op);
        }
        delete_nodes.insert(cur_slice);
        delete_nodes.insert(cur_slice_out);
        delete_counts++;
      }
      GraphSafeRemoveNodes(graph, delete_nodes);
    }
  };

  gpd(graph, handler);
  return delete_counts;
}

void DeleteRepeatedOpsPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int delete_counts = DeleteShapePass(graph);
  if (delete_counts > 0) {
    LOG(INFO) << "--- delete " << delete_counts << " repeated shape ops";
  }

  delete_counts = DeleteSlicePass(graph);
  if (delete_counts > 0) {
    LOG(INFO) << "--- delete " << delete_counts << " repeated slice ops";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_repeated_ops_pass,
              paddle::framework::ir::DeleteRepeatedOpsPass);

REGISTER_PASS_CAPABILITY(delete_repeated_ops_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "shape", 0));
