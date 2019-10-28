/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/elementwise_group_detector.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

static std::unordered_set<std::string> binary_op_types = {
    "elementwise_add", "elementwise_sub", "elementwise_mul",
    "elementwise_div", "elementwise_min", "elementwise_max"};

static std::unordered_set<std::string> unary_op_types = {"relu", "sigmoid",
                                                         "tanh"};

static bool IsSpecifiedOp(const std::unordered_set<std::string>& op_types,
                          Node* n) {
  if (n && n->IsOp() && n->Op() && n->outputs.size() > 0U) {
    auto iter = op_types.find(n->Op()->Type());
    if (iter != op_types.end()) {
      return true;
    }
  }
  return false;
}

static bool IsBinaryOp(Node* n) {
  if (IsSpecifiedOp(binary_op_types, n) && n->inputs.size() == 2U) {
    auto* x = n->inputs[0];
    auto* y = n->inputs[1];

    std::vector<int64_t> x_shape;
    std::vector<int64_t> y_shape;
    if (x && x->IsVar() && x->Var()) {
      x_shape = x->Var()->GetShape();
    }
    if (y && y->IsVar() && y->Var()) {
      y_shape = y->Var()->GetShape();
    }
    if (x_shape.size() == 0U || x_shape.size() != y_shape.size()) {
      return false;
    }
    for (size_t i = 0; i < x_shape.size(); ++i) {
      if (x_shape[i] != y_shape[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

static bool IsUnaryOp(Node* n) { return IsSpecifiedOp(unary_op_types, n); }

bool ElementwiseGroupDetector::IsElementwiseOp(Node* n) {
  return IsBinaryOp(n) || IsUnaryOp(n);
}

bool ElementwiseGroupDetector::IsInputOfElementwiseOp(Node* n,
                                                      std::string name) {
  if (n && n->IsVar() && n->Var()) {
    for (auto* op : n->outputs) {
      if (IsElementwiseOp(op)) {
        if (name.empty()) {
          return true;
        } else if (IsNthInput(n, op, name, 0)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool ElementwiseGroupDetector::IsOutputOfElementwiseOp(Node* n) {
  if (n && n->IsVar() && n->Var()) {
    for (auto* op : n->inputs) {
      if (IsElementwiseOp(op)) {
        return true;
      }
    }
  }
  return false;
}

void ElementwiseGroupDetector::Insert(Node* n) {
  if (subgraph_.nodes_set.find(n) == subgraph_.nodes_set.end()) {
    VLOG(5) << "Insert " << n->Name() << " to subgraph " << name_;
    subgraph_.nodes_set.insert(n);
  }
}

int ElementwiseGroupDetector::Search(Node* n, std::vector<Node*> except_nodes) {
  std::unordered_set<Node*> except_nodes_set;
  for (size_t i = 0; i < except_nodes.size(); ++i) {
    except_nodes_set.insert(except_nodes[i]);
  }

  int num_operations = 0;
  if (IsElementwiseOp(n)) {
    Insert(n);
    num_operations += 1;
    for (auto* var : n->inputs) {
      Insert(var);
      if (except_nodes_set.find(var) == except_nodes_set.end()) {
        num_operations += Search(var, {n});
      }
    }
    for (auto* var : n->outputs) {
      Insert(var);
      if (except_nodes_set.find(var) == except_nodes_set.end()) {
        num_operations += Search(var, {n});
      }
    }
  } else if (n && n->IsVar() && n->Var()) {
    for (auto* op : n->inputs) {
      if (IsElementwiseOp(op) &&
          except_nodes_set.find(op) == except_nodes_set.end()) {
        num_operations += Search(op, {n});
      }
    }
    for (auto* op : n->outputs) {
      if (IsElementwiseOp(op) &&
          except_nodes_set.find(op) == except_nodes_set.end()) {
        num_operations += Search(op, {n});
      }
    }
  }
  return num_operations;
}

int ElementwiseGroupDetector::operator()(Node* n) {
  if (!IsOutputOfElementwiseOp(n) && IsInputOfElementwiseOp(n, "X")) {
    name_ = n->Name();
    Insert(n);
    num_operations_ = Search(n, n->inputs);
    VLOG(4) << "Detect elementwise subgraph begin with " << name_ << ", "
            << num_operations_ << " operations, " << GetSubgraph().GetNumNodes()
            << " nodes";
  }
  return num_operations_;
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
