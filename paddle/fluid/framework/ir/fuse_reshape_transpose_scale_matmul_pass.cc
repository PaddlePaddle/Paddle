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

#include "paddle/fluid/framework/ir/fuse_reshape_transpose_scale_matmul_pass.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

// Get input/output node via the name.
inline Node* GetNode(const Node* node, bool is_out, std::string type = "X") {
  if (is_out) {
    for (auto it = node->outputs.begin(); it != node->outputs.end(); it++) {
      if (0 == node->Op()->Output(type)[0].compare((*it)->Name())) {
        return *it;
      }
    }
  } else {
    for (auto it = node->inputs.begin(); it != node->inputs.end(); it++) {
      if (0 == node->Op()->Input(type)[0].compare((*it)->Name())) {
        return *it;
      }
    }
  }

  return nullptr;
}

// Get input/output node via the index.
inline Node* GetNode(const Node* node, bool is_out, size_t index) {
  if (is_out) {
    if (node->outputs.size() > index) {
      return node->outputs[index];
    }
  } else {
    if (node->inputs.size() > index) {
      return node->inputs[index];
    }
  }

  return nullptr;
}

void ReshapeTransposeScaleMatmulFusePass::GetSpeicalOpNodes(
    const std::vector<Node*>& nodes, std::string type,
    std::vector<Node*>* dst_nodes) const {
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    auto node = *it;
    if (node->IsOp() && (!node->Name().compare(type))) {
      dst_nodes->push_back(node);
    }
  }
}

void ReshapeTransposeScaleMatmulFusePass::UpdateFusedNode(
    ir::Graph* graph, Node* matmul_op, std::vector<Node*>& nodes) const {
  std::unordered_set<const Node*> remove_nodes;
  float bias = 0.0f, alpha = 1.0f;

  auto matmul_input = nodes[MATMUL_INPUT],
       transpose_input = nodes[TRANSPOSE_INPUT],
       reshape_input = nodes[RESHAPE_INPUT];
  auto reshape_op = nodes[RESHAPE_OP], transpose_op = nodes[TRANSPOSE_OP];
  Node *scale_op = nullptr, *scale_input = nullptr;
  Node* reshape_output = nullptr;
  auto matmul_output = matmul_input;

  // Check if the node is the output of matmul operator.
  bool is_out = matmul_output == GetNode(matmul_op, true, "Out");

  if (is_out) {
    // Reconfigure the relevant nodes.
    transpose_input = nodes[TRANSPOSE_REVERSE_INPUT];
    transpose_op = nodes[TRANSPOSE_REVERSE_OP];
    reshape_input = nodes[RESHAPE_REVERSE_INPUT];
    reshape_op = nodes[RESHAPE_REVERSE_OP];
    reshape_output = nodes[RESHAPE_REVERSE_OUTPUT];
  } else {
    // To configue the scale and scale's input node.
    if (nodes.size() == MAX_MATMUL_NODES) {
      scale_op = nodes[SCALE_OP];
      scale_input = nodes[SCALE_INPUT];
      matmul_input = nodes[SCALE_OUTPUT];
    }
  }

  // Check if the node is the "X" input node of matmul operator.
  bool is_x = matmul_input == GetNode(matmul_op, false, "X");

  auto reshape_shape_tz =
      boost::get<std::vector<int>>(reshape_op->Op()->GetAttr("shape"));
  auto transpose_axis_tz =
      boost::get<std::vector<int>>(transpose_op->Op()->GetAttr("axis"));

  // To set the shape and axis attributes of matmul operator.
  if (is_out) {
    matmul_op->Op()->SetAttr("shape_Out", reshape_shape_tz);
    matmul_op->Op()->SetAttr("axis_Out", transpose_axis_tz);
  } else {
    if (is_x) {
      matmul_op->Op()->SetAttr("shape_X", reshape_shape_tz);
      matmul_op->Op()->SetAttr("axis_X", transpose_axis_tz);
    } else {
      matmul_op->Op()->SetAttr("shape_Y", reshape_shape_tz);
      matmul_op->Op()->SetAttr("axis_Y", transpose_axis_tz);
    }
  }

  if (is_out && reshape_output != nullptr) {
    reshape_output->inputs.clear();
    reshape_op->outputs.clear();

    matmul_op->outputs.clear();
    matmul_output->inputs.clear();

    matmul_op->Op()->SetOutput(
        "Out", std::vector<std::string>({reshape_output->Name()}));

    IR_NODE_LINK_TO(matmul_op, reshape_output);
    remove_nodes.insert(
        {reshape_op, transpose_op, transpose_input, reshape_input});

  } else {
    reshape_input->outputs.clear();
    reshape_op->inputs.clear();

    // Get the bias and scale value of scale operator.
    if (scale_op != nullptr) {
      bias = boost::get<float>(scale_op->Op()->GetAttr("bias"));
      alpha = boost::get<float>(scale_op->Op()->GetAttr("scale"));
    }

    // If the bias value is not equal to zero, the scale operator need be keep
    // and change the input nodes. When the bias value is equal to zero, only
    // configure the scale as the alpha value of matmul operator and remove the
    // scale operator.
    if (scale_op != nullptr && scale_input != nullptr && bias != 0.0f) {
      scale_input->outputs.clear();
      scale_op->inputs.clear();
      scale_op->Op()->SetInput(
          "X", std::vector<std::string>({reshape_input->Name()}));
      IR_NODE_LINK_TO(reshape_input, scale_op);

      remove_nodes.insert(
          {reshape_op, transpose_op, transpose_input, scale_input});
    } else {
      if (scale_op != nullptr && scale_input != nullptr) {
        matmul_op->Op()->SetAttr("alpha", alpha);
        remove_nodes.insert({scale_op, scale_input});
      }
      matmul_input->outputs.clear();

      for (auto it = matmul_op->inputs.begin();
           it != matmul_op->inputs.end();) {
        if (*it == matmul_input) {
          it = matmul_op->inputs.erase(it);
        } else {
          it++;
        }
      }
      if (is_x) {
        matmul_op->Op()->SetInput(
            "X", std::vector<std::string>({reshape_input->Name()}));
      } else {
        matmul_op->Op()->SetInput(
            "Y", std::vector<std::string>({reshape_input->Name()}));
      }

      IR_NODE_LINK_TO(reshape_input, matmul_op);
      remove_nodes.insert(
          {reshape_op, transpose_op, transpose_input, matmul_input});
    }
  }

  // Remove all unused nodes.
  GraphSafeRemoveNodes(graph, remove_nodes);
}

int ReshapeTransposeScaleMatmulFusePass::ReConfigureMatMulOp(
    ir::Graph* graph,
    std::multimap<Node*, std::vector<Node*>>& matmul_nodes_map) const {
  int count = 0;

  Node* node = nullptr;
  for (auto it = matmul_nodes_map.begin(), it_start = it;
       it != matmul_nodes_map.end();) {
    node = it->first;
    it++;
    if (it == matmul_nodes_map.end() || node != it->first) {
      while (it_start != it) {
        // Update the attributes and input/output nodes of the fused matmul
        // node.
        UpdateFusedNode(graph, node, it_start->second);
        count++;
        it_start++;
      }
    }
  }

  return count;
}

bool ReshapeTransposeScaleMatmulFusePass::IsEnableFuse(
    std::vector<Node*>& nodes, bool is_out) const {
  auto transpose_op = nodes[TRANSPOSE_OP];

  bool ret = transpose_op && transpose_op->IsOp() &&
             transpose_op->Op()->Type() == "transpose2" &&
             transpose_op->Op()->HasAttr("axis");

  if (!ret) {
    return false;
  }

  auto transpose_axis_tz =
      boost::get<std::vector<int>>(transpose_op->Op()->GetAttr("axis"));

  int length = transpose_axis_tz.size();
  if (transpose_axis_tz[length - 1] != length - 1) {
    return false;
  }

  if (is_out) {
    if ((transpose_axis_tz.size() == 4) &&
        (transpose_axis_tz[length - 2] != length - 3 ||
         transpose_axis_tz[length - 3] != length - 2)) {
      return false;
    }
  }

  return true;
}

// To detect the subgraphs to fuse the transpose and reshape operators.
int ReshapeTransposeScaleMatmulFusePass::DetectFuseNodes(
    const std::vector<Node*>& matmul_nodes,
    std::multimap<Node*, std::vector<Node*>>& matmul_nodes_map) const {
  for (auto matmul_node : matmul_nodes) {
    for (auto var_name : {"X", "Y", "Out"}) {
      std::vector<Node*> nodes;

      std::vector<std::pair<std::string, std::string>> map;
      map.push_back(std::make_pair("matmul", var_name));
      map.push_back(std::make_pair("scale", "X"));
      map.push_back(std::make_pair("transpose2", "X"));
      map.push_back(std::make_pair("reshape2", "X"));

      bool is_out = !strcmp(var_name, "Out");
      Node *op_node = matmul_node, *var_node;

      for (auto it = map.begin(); it != map.end();) {
        if (op_node == nullptr) {
          nodes.clear();
          break;
        }
        if (op_node->Op()->Type() != it->first) {
          if (it->first == "scale") {
            it++;
            if (op_node->Op()->Type() != it->first) {
              nodes.clear();
              break;
            }
          } else {
            nodes.clear();
            break;
          }
        }
        if (is_out) {
          var_node = GetNode(op_node, is_out, 0);
        } else {
          var_node = GetNode(op_node, is_out, it->second);
        }
        op_node = GetNode(var_node, is_out, 0);
        it++;
        if (it == map.end()) {
          nodes.insert(nodes.begin(), var_node);
          break;
        } else {
          nodes.insert(nodes.begin(), var_node);
          nodes.insert(nodes.begin(), op_node);
        }
      }
      if (nodes.size() >= MAX_MATMUL_NODES - 2) {
        if (IsEnableFuse(nodes, is_out)) {
          matmul_nodes_map.insert(std::make_pair(matmul_node, nodes));
        }
      }
    }
  }

  return matmul_nodes_map.size();
}

void ReshapeTransposeScaleMatmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE(graph);
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

  // To obtain all operator nodes.
  std::vector<Node*> nodes = TopologySortOperations(*graph);

  // To obtain all matmul nodes.
  std::vector<Node*> matmul_nodes;
  GetSpeicalOpNodes(nodes, "matmul", &matmul_nodes);

  // To detect all subgraphs that it can fuse transpose, reshape and matmul
  // operators.
  std::multimap<Node*, std::vector<Node*>> matmul_nodes_map;
  auto found_fuse_count = DetectFuseNodes(matmul_nodes, matmul_nodes_map);

  if (found_fuse_count > 0) {
    std::cout << "---  detect " << found_fuse_count << " subgraphs"
              << std::endl;

    // To reconfigure the matmul operators.
    found_fuse_count = ReConfigureMatMulOp(graph, matmul_nodes_map);

    std::cout << "Fused graph " << found_fuse_count << std::endl;
  }

  AddStatis(found_fuse_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_reshape_transpose_scale_matmul_pass,
              paddle::framework::ir::ReshapeTransposeScaleMatmulFusePass);
