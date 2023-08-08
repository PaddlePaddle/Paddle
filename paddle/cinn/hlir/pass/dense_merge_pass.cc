// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/hlir/pass/fusion_helper_base.h"

namespace cinn {
namespace hlir {
namespace pass {

using common::GraphNode;
using framework::Graph;
using framework::Node;
using framework::NodeAttr;

// Dense Merge Pass: merge those gemm which has same var as input into a batched
// cubals call op. A * B, A * C, A * D,... after A * [B, C, D,...] Using cublas
// batched gemm can avoid do concat and slice.

class DenseMergePassHelper : public FusionHelperBase {
 public:
  explicit DenseMergePassHelper(Graph* graph)
      : FusionHelperBase(graph), graph_(graph) {}

  void operator()() {
    auto nodes_inorder = std::get<0>(graph_->topological_order());
    for (auto node : nodes_inorder) {
      if (removed_node_set_.count(node)) {
        continue;
      }
      if (node->safe_as<NodeData>()) {
        MergeDense(node->safe_as<NodeData>());
      }
    }
  }

 private:
  void MergeDense(NodeData* node) {
    auto dense_ops = GetDenseOp(node);
    if (dense_ops.size() <= 1) {
      return;
    }

    std::vector<Node*> lhs_ops, rhs_ops;
    for (auto op : dense_ops) {
      const auto& in_links = op->inlinks_in_order();
      CHECK(!in_links.empty());
      if (in_links[0]->source() == node) {
        lhs_ops.push_back(op);
      } else {
        rhs_ops.push_back(op);
      }
    }

    if (lhs_ops.size() > 1) LeftMerge(node, lhs_ops);
    if (rhs_ops.size() > 1) RightMerge(node, rhs_ops);
  }

  std::vector<Node*> GetDenseOp(NodeData* node) {
    std::vector<Node*> dense_ops;
    for (auto link : node->outlinks()) {
      auto sink = link->sink()->safe_as<Node>();
      if (sink->op()->name == "matmul" || sink->op()->name == "mul" ||
          sink->op()->name == "cublas_gemm" ||
          sink->op()->name == "cublas_matmul") {
        if (std::find(dense_ops.begin(), dense_ops.end(), sink) ==
            dense_ops.end()) {
          dense_ops.push_back(sink);
        }
      }
    }
    return dense_ops;
  }

  void LeftMerge(NodeData* node, std::vector<Node*> dense_ops) {
    DoMerge(node, dense_ops, 1, "left");
  }

  void RightMerge(NodeData* node, std::vector<Node*> dense_ops) {
    DoMerge(node, dense_ops, 0, "right");
  }

  void DoMerge(NodeData* node,
               std::vector<Node*> dense_ops,
               int pos,
               std::string side) {
    // split dense op by it's attr
    std::unordered_map<std::string, std::vector<Node*>> dense_op_map;
    for (auto dense_op : dense_ops) {
      const auto& in_links = dense_op->inlinks_in_order();
      CHECK_GT(in_links.size(), pos);
      auto sign = GenOpSign(in_links[pos]->source()->safe_as<NodeData>(),
                            dense_op->attrs);
      if (dense_op_map.count(sign)) {
        dense_op_map[sign].push_back(dense_op);
      } else {
        dense_op_map[sign] = {dense_op};
      }
    }

    for (auto dense_op : dense_op_map) {
      if (dense_op.second.size() <= 1) {
        continue;
      }

      // create custom call node
      Node* node_tmp = new Node(Operator::Get("custom_call"),
                                "custom_call",
                                common::UniqName("custom_call"));
      graph_->RegisterNode(node_tmp->id(), node_tmp);
      node_tmp->attrs.attr_store = dense_op.second[0]->attrs.attr_store;
      node_tmp->attrs.attr_store["side"] = side;
      node_tmp->attrs.attr_store["custom_call"] =
          std::string("cinn_call_batched_cublas");

      // update inlink.
      node->LinkTo(node_tmp);
      for (auto op : dense_op.second) {
        const auto& in_links = op->inlinks_in_order();
        node->UnLinkSingleTo(op);
        // link to new node
        CHECK_GT(in_links.size(), pos);
        in_links[pos]->source()->LinkTo(node_tmp);
        // unlink old dense node
        in_links[pos]->source()->UnLinkSingleTo(op);
        // dense_node_data link to node_tmp
        auto op_node_data = GetNodeData(op);
        op->UnLinkSingleTo(op_node_data);
        node_tmp->LinkTo(op_node_data);
        // update node tmp.
        op_node_data->source_node.Reset(node_tmp);

        removed_node_set_.insert(op);
        graph_->DropNode(op);
      }
    }
  }

  std::string GenOpSign(const NodeData* node, const NodeAttr& attrs) {
    auto attr_store = attrs.attr_store;
    bool trans_a = attr_store.count("trans_a")
                       ? absl::get<bool>(attr_store.at("trans_a"))
                       : false;
    bool trans_b = attr_store.count("trans_b")
                       ? absl::get<bool>(attr_store.at("trans_b"))
                       : false;
    bool trans_out = attr_store.count("trans_out")
                         ? absl::get<bool>(attr_store.at("trans_out"))
                         : false;
    float alpha = attr_store.count("alpha")
                      ? absl::get<float>(attr_store.at("alpha"))
                      : 1.0f;
    float beta = attr_store.count("beta")
                     ? absl::get<float>(attr_store.at("beta"))
                     : 0.0f;
    int x_num_col_dims = attr_store.count("x_num_col_dims")
                             ? absl::get<int>(attr_store.at("x_num_col_dims"))
                             : 0;
    int y_num_col_dims = attr_store.count("y_num_col_dims")
                             ? absl::get<int>(attr_store.at("y_num_col_dims"))
                             : 0;

    std::string sign = "";
    sign += std::to_string(trans_a);
    sign += "_" + std::to_string(trans_b);
    sign += "_" + std::to_string(trans_out);
    sign += "_" + std::to_string(alpha);
    sign += "_" + std::to_string(beta);
    sign += "_" + std::to_string(x_num_col_dims);
    sign += "_" + std::to_string(y_num_col_dims);
    auto shape = shape_dict_.at(node->id());
    for (auto s : shape) {
      sign += "_" + std::to_string(s);
    }

    return sign;
  }

 private:
  std::unordered_set<GraphNode*> removed_node_set_;
  Graph* graph_;
};

void DenseMergePassInternal(Graph* graph) {
  DenseMergePassHelper dense_merge_pass_helper(graph);
  dense_merge_pass_helper();
}

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(DenseMergePass) {
  CINN_REGISTER_PASS(DenseMergePass)
      .describe("")
      .set_change_structure(true)
      .provide_graph_attr("infershape")
      .provide_graph_attr("inferdtype")
      .set_body(cinn::hlir::pass::DenseMergePassInternal);
  return true;
}
