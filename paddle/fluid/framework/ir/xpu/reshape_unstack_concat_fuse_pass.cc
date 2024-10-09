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

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
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

struct ReshapeUnstackConcatPattern : public PatternBase {
  ReshapeUnstackConcatPattern(PDPattern* pattern,
                              const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(reshape);
  PATTERN_DECL_NODE(unstack);
  PATTERN_DECL_NODE(concat);
  // declare variable node's name
  PATTERN_DECL_NODE(reshape_in);
  PATTERN_DECL_NODE(reshape_out);
  PATTERN_DECL_NODE(unstack_out0);
  PATTERN_DECL_NODE(concat_out);
};

ReshapeUnstackConcatPattern::ReshapeUnstackConcatPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* reshape_in =
      pattern->NewNode(reshape_in_repr())->assert_is_op_input("reshape2", "X");
  auto* reshape =
      pattern->NewNode(reshape_repr())
          ->assert_is_op("reshape2")
          ->assert_more([](Node* node) {
            auto shape = node->Op()->GetAttrIfExists<std::vector<int>>("shape");
            return shape.size() == 6;
          });
  auto* reshape_out = pattern->NewNode(reshape_out_repr())
                          ->assert_is_op_output("reshape2", "Out")
                          ->assert_is_op_input("unstack", "X");
  auto* unstack = pattern->NewNode(unstack_repr())
                      ->assert_is_op("unstack")
                      ->assert_more([](Node* node) {
                        auto axis = node->Op()->GetAttrIfExists<int>("axis");
                        return axis == 0;
                      });
  auto* unstack_out0 = pattern->NewNode(unstack_out0_repr())
                           ->assert_is_op_nth_output("unstack", "Y", 0)
                           ->assert_is_op_nth_input("concat", "X", 0);
  auto* concat = pattern->NewNode(concat_repr())
                     ->assert_is_op("concat")
                     ->assert_more([](Node* node) {
                       auto axis = node->Op()->GetAttrIfExists<int>("axis");
                       return axis == -2;
                     });
  auto* concat_out = pattern->NewNode(concat_out_repr())
                         ->assert_is_op_output("concat", "Out")
                         ->assert_more([](Node* node) {
                           auto out_nodes = node->outputs;
                           if (out_nodes.size() <= 1) {
                             return false;
                           }
                           for (auto out_node : out_nodes) {
                             if (out_node->Name() != "slice") {
                               return false;
                             }
                           }
                           return true;
                         });
  reshape->LinksFrom({reshape_in}).LinksTo({reshape_out});
  unstack->LinksFrom({reshape_out}).LinksTo({unstack_out0});
  concat->LinksFrom({unstack_out0}).LinksTo({concat_out});
}

}  // namespace patterns

class ReshapeUnstackConcatFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"reshape_unstack_concat_fuse_pass"};
};

// clang-format off
/*
Origin subgraph:
                                  reshape(4,-1,48,2,16,4096)
                                    |
                                  unstack
                                    |
                                  concat
                                    |
    ------------------------------------------------------------------
    |                               |                                |
slice(start/end/axes:0/1/1)    slice(start/end/axes:1/2/1)   ...   slice(start/end/axes:n-1/n/1)
    |                               |                                |
reshape(-1,2,64,4,1024)        reshape(-1,2,64,4,1024)       ...   reshape(-1,2,64,4,1024)
    |                               |                                |
slice(start/end/axes:0/1/3)    slice(start/end/axes:0/1/3)   ...   slice(start/end/axes:0/1/3)
    |                               |                                |
reshape(-1,2,64,16,64)         reshape(-1,2,64,16,64)        ...   reshape(-1,2,64,16,64)
    |                               |                                |
transpose(1,0,3,2,4)           transpose(1,0,3,2,4)          ...   transpose(1,0,3,2,4)

Optimized subgraph:
                                  reshape(-1,4,1024)
                                    |
                                  slice(start/end/axes:0/1/2)
                                    |
                                  reshape(4,-1,48,2,16,1024)
                                    |
                                  unstack
                                    |
                                  concat
                                    |
                                  reshape(-1,n*2,64,16,64)
                                    |
                                  transpose(1,0,3,2,4)
                                    |
                                  split(num/axis:n/0)
                                    |
    ------------------------------------------------------------------
    |                               |                                |
*/
// clang-format on
void ReshapeUnstackConcatFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::ReshapeUnstackConcatPattern pattern(gpd.mutable_pattern(),
                                                name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ReshapeUnstackConcatFusePass fuse";
    GET_IR_NODE(reshape);
    GET_IR_NODE(unstack);
    GET_IR_NODE(concat);
    GET_IR_NODE(reshape_in);
    GET_IR_NODE(reshape_out);
    GET_IR_NODE(unstack_out0);
    GET_IR_NODE(concat_out);
    auto* block = reshape->Op()->Block();

    auto concat_out_ops = concat_out->outputs;
    int ops_num = concat_out_ops.size();
    std::vector<Node*> slice_0s(ops_num, nullptr);
    std::vector<Node*> reshape_0s(ops_num, nullptr);
    std::vector<Node*> slice_1s(ops_num, nullptr);
    std::vector<Node*> reshape_1s(ops_num, nullptr);
    std::vector<Node*> transposes(ops_num, nullptr);
    for (int i = 0; i < ops_num; i++) {
      auto slice_0 = concat_out_ops[i];
      if (slice_0->Name() != "slice") return;
      auto slice_0_starts =
          slice_0->Op()->GetAttrIfExists<std::vector<int>>("starts");
      auto slice_0_ends =
          slice_0->Op()->GetAttrIfExists<std::vector<int>>("ends");
      auto slice_0_axes =
          slice_0->Op()->GetAttrIfExists<std::vector<int>>("axes");
      if (slice_0_starts.size() != 1 ||
          (slice_0_ends[0] - slice_0_starts[0] != 1) || slice_0_axes[0] != 1) {
        return;
      }
      int op_index = slice_0_starts[0];
      if (slice_0s[op_index] != nullptr) return;
      slice_0s[op_index] = slice_0;

      auto reshape_0 = slice_0->outputs[0]->outputs[0];
      if (reshape_0->Name() != "reshape2") return;
      auto reshape_0_shape =
          reshape_0->Op()->GetAttrIfExists<std::vector<int>>("shape");
      if (reshape_0_shape.size() != 5) return;
      reshape_0s[op_index] = reshape_0;

      Node* slice_1 = nullptr;
      for (auto reshape_out : reshape_0->outputs) {
        if (reshape_out->Name() == reshape_0->Op()->Output("Out")[0]) {
          slice_1 = reshape_out->outputs[0];
          if (slice_1->Name() != "slice") return;
          auto slice_1_axes =
              slice_1->Op()->GetAttrIfExists<std::vector<int>>("axes");
          if (slice_1_axes.size() != 1 || slice_1_axes[0] != 3) {
            return;
          }
          slice_1s[op_index] = slice_1;
        }
      }

      auto* reshape_1 = slice_1->outputs[0]->outputs[0];
      if (reshape_1->Name() != "reshape2") return;
      auto reshape_1_shape =
          reshape_1->Op()->GetAttrIfExists<std::vector<int>>("shape");
      if (reshape_1_shape.size() != 5) return;
      reshape_1s[op_index] = reshape_1;

      Node* transpose = nullptr;
      for (auto reshape_out : reshape_1->outputs) {
        if (reshape_out->Name() == reshape_1->Op()->Output("Out")[0]) {
          transpose = reshape_out->outputs[0];
          if (transpose->Name() != "transpose2") return;
          auto transpose_axis =
              transpose->Op()->GetAttrIfExists<std::vector<int>>("axis");
          if (transpose_axis != std::vector<int>{1, 0, 3, 2, 4}) return;
          transposes[op_index] = transpose;
        }
      }
    }

    std::string new_reshape_0_out_name = reshape_in->Name() + "_reshape_out";
    VarDesc new_reshape_0_out_desc(new_reshape_0_out_name);
    Node* new_reshape_0_out = graph->CreateVarNode(&new_reshape_0_out_desc);

    framework::OpDesc new_reshape_0_op_desc(block);
    new_reshape_0_op_desc.SetType("reshape2");
    auto reshape_0_shape =
        reshape_0s[0]->Op()->GetAttrIfExists<std::vector<int>>("shape");
    std::vector<int> new_reshape_0_shape{
        -1, reshape_0_shape[3], reshape_0_shape[4]};
    new_reshape_0_op_desc.SetAttr("shape", new_reshape_0_shape);
    new_reshape_0_op_desc.SetInput("X", {reshape_in->Name()});
    new_reshape_0_op_desc.SetOutput("Out", {new_reshape_0_out_name});
    auto* new_reshape_0 = graph->CreateOpNode(&new_reshape_0_op_desc);

    std::string new_slice_0_out_name = reshape_in->Name() + "_slice_out";
    VarDesc new_slice_0_out_desc(new_slice_0_out_name);
    Node* new_slice_0_out = graph->CreateVarNode(&new_slice_0_out_desc);

    framework::OpDesc new_slice_0_op_desc(block);
    new_slice_0_op_desc.SetType("slice");
    auto new_slice_0_start =
        slice_1s[0]->Op()->GetAttrIfExists<std::vector<int>>("starts");
    auto new_slice_0_ends =
        slice_1s[0]->Op()->GetAttrIfExists<std::vector<int>>("ends");
    new_slice_0_op_desc.SetAttr("starts", new_slice_0_start);
    new_slice_0_op_desc.SetAttr("ends", new_slice_0_ends);
    new_slice_0_op_desc.SetAttr("axes", std::vector<int>{1});
    new_slice_0_op_desc.SetAttr("decrease_axis", std::vector<int>{1});
    new_slice_0_op_desc.SetInput("Input", {new_reshape_0_out_name});
    new_slice_0_op_desc.SetOutput("Out", {new_slice_0_out_name});
    auto* new_slice_0 = graph->CreateOpNode(&new_slice_0_op_desc);

    reshape->Op()->SetInput("X", {new_slice_0_out_name});
    auto reshape_shape =
        reshape->Op()->GetAttrIfExists<std::vector<int>>("shape");
    reshape_shape[5] /= reshape_0_shape[3];
    reshape->Op()->SetAttr("shape", reshape_shape);
    IR_NODE_UNLINK(reshape_in, reshape);
    IR_NODE_LINK_TO(reshape_in, new_reshape_0);
    IR_NODE_LINK_TO(new_reshape_0, new_reshape_0_out);
    IR_NODE_LINK_TO(new_reshape_0_out, new_slice_0);
    IR_NODE_LINK_TO(new_slice_0, new_slice_0_out);
    IR_NODE_LINK_TO(new_slice_0_out, reshape);

    std::string new_reshape_1_out_name = concat_out->Name() + "_reshape_out";
    VarDesc new_reshape_1_out_desc(new_reshape_1_out_name);
    Node* new_reshape_1_out = graph->CreateVarNode(&new_reshape_1_out_desc);

    framework::OpDesc new_reshape_1_op_desc(block);
    new_reshape_1_op_desc.SetType("reshape2");
    auto new_reshape_1_shape =
        reshape_1s[0]->Op()->GetAttrIfExists<std::vector<int>>("shape");
    new_reshape_1_shape[1] *= ops_num;
    new_reshape_1_op_desc.SetAttr("shape", new_reshape_1_shape);
    new_reshape_1_op_desc.SetInput("X", {concat_out->Name()});
    new_reshape_1_op_desc.SetOutput("Out", {new_reshape_1_out_name});
    auto* new_reshape_1 = graph->CreateOpNode(&new_reshape_1_op_desc);

    std::string new_transpose_0_out_name =
        concat_out->Name() + "_transpose_out";
    VarDesc new_transpose_0_out_desc(new_transpose_0_out_name);
    Node* new_transpose_0_out = graph->CreateVarNode(&new_transpose_0_out_desc);

    framework::OpDesc new_transpose_0_op_desc(block);
    new_transpose_0_op_desc.SetType("transpose2");
    auto transpose_axis =
        transposes[0]->Op()->GetAttrIfExists<std::vector<int>>("axis");
    new_transpose_0_op_desc.SetAttr("axis", transpose_axis);
    new_transpose_0_op_desc.SetInput("X", {new_reshape_1_out_name});
    new_transpose_0_op_desc.SetOutput("Out", {new_transpose_0_out_name});
    auto* new_transpose_0 = graph->CreateOpNode(&new_transpose_0_op_desc);

    std::vector<std::string> new_split_0_out_names;
    for (auto* transpose : transposes) {
      new_split_0_out_names.push_back(transpose->Op()->Output("Out")[0]);
    }

    framework::OpDesc new_split_0_op_desc(block);
    new_split_0_op_desc.SetType("split");
    new_split_0_op_desc.SetAttr("num", ops_num);
    new_split_0_op_desc.SetAttr("axis", 0);
    new_split_0_op_desc.SetInput("X", {new_transpose_0_out_name});
    new_split_0_op_desc.SetOutput("Out", new_split_0_out_names);
    auto* new_split_0 = graph->CreateOpNode(&new_split_0_op_desc);

    IR_NODE_LINK_TO(concat_out, new_reshape_1);
    IR_NODE_LINK_TO(new_reshape_1, new_reshape_1_out);
    IR_NODE_LINK_TO(new_reshape_1_out, new_transpose_0);
    IR_NODE_LINK_TO(new_transpose_0, new_transpose_0_out);
    IR_NODE_LINK_TO(new_transpose_0_out, new_split_0);
    for (auto* transpose : transposes) {
      for (auto* transpose_out : transpose->outputs) {
        if (transpose_out->Name() == transpose->Op()->Output("Out")[0]) {
          IR_NODE_LINK_TO(new_split_0, transpose_out);
        }
      }
    }

    std::unordered_set<const Node*> delete_nodes;
    delete_nodes.insert(slice_0s.begin(), slice_0s.end());
    for (auto* slice_0 : slice_0s) {
      delete_nodes.emplace(slice_0->outputs[0]);
    }
    delete_nodes.insert(reshape_0s.begin(), reshape_0s.end());
    for (auto* reshape_0 : reshape_0s) {
      auto reshape_0_outs = reshape_0->outputs;
      delete_nodes.insert(reshape_0_outs.begin(), reshape_0_outs.end());
    }
    delete_nodes.insert(slice_1s.begin(), slice_1s.end());
    for (auto* slice_1 : slice_1s) {
      delete_nodes.emplace(slice_1->outputs[0]);
    }
    delete_nodes.insert(reshape_1s.begin(), reshape_1s.end());
    for (auto* reshape_1 : reshape_1s) {
      auto reshape_1_outs = reshape_1->outputs;
      delete_nodes.insert(reshape_1_outs.begin(), reshape_1_outs.end());
    }
    delete_nodes.insert(transposes.begin(), transposes.end());
    GraphSafeRemoveNodes(graph, delete_nodes);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reshape_unstack_concat_fuse_pass,
              paddle::framework::ir::ReshapeUnstackConcatFusePass);

REGISTER_PASS_CAPABILITY(reshape_unstack_concat_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "stack", 0));
