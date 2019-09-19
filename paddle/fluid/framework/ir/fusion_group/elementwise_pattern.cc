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

#include "paddle/fluid/framework/ir/fusion_group/elementwise_pattern.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

static std::unordered_set<std::string> binary_op_types = {
    "elementwise_add", "elementwise_sub", "elementwise_mul",
    "elementwise_div", "elementwise_min", "elementwise_max"};

static std::unordered_set<std::string> unary_op_types = {"relu", "sigmoid"};

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

bool IsElementwiseOp(Node* n) { return IsBinaryOp(n) || IsUnaryOp(n); }

bool IsInputOfElementwiseOp(Node* n) {
  if (n && n->IsVar() && n->Var()) {
    for (auto* op : n->outputs) {
      if (IsElementwiseOp(op)) {
        return true;
      }
    }
  }
  return false;
}

bool IsOutputOfElementwiseOp(Node* n) {
  if (n && n->IsVar() && n->Var()) {
    for (auto* op : n->inputs) {
      if (IsElementwiseOp(op)) {
        return true;
      }
    }
  }
  return false;
}

static bool IsInputOfRepeatedElementwiseOp(Node* n, int num) {
  Node* p = n;
  for (int i = 0; i < num; ++i) {
    bool find = false;
    if (p && p->IsVar() && p->Var()) {
      for (auto* op : p->outputs) {
        if (IsElementwiseOp(op)) {
          p = op->outputs[0];
          find = true;
          break;
        }
      }
    }
    if (!find) {
      return false;
    }
  }
  return true;
}

int NumAbjacentElementwiseOps(Node* n, std::vector<Node*> expect_nodes) {
  // LOG(INFO) << "Enter NumAbjacentElementwiseOps: " << n->Name();
  std::unordered_set<Node*> expect_nodes_set;
  for (size_t i = 0; i < expect_nodes.size(); ++i) {
    expect_nodes_set.insert(expect_nodes[i]);
  }

  int res = 0;
  if (IsElementwiseOp(n)) {
    res += 1;
    for (auto* var : n->inputs) {
      if (expect_nodes_set.find(var) == expect_nodes_set.end()) {
        res += NumAbjacentElementwiseOps(var, {n});
      }
    }
    for (auto* var : n->outputs) {
      if (expect_nodes_set.find(var) == expect_nodes_set.end()) {
        res += NumAbjacentElementwiseOps(var, {n});
      }
    }
  } else if (n && n->IsVar() && n->Var()) {
    for (auto* op : n->inputs) {
      if (IsElementwiseOp(op) &&
          expect_nodes_set.find(op) == expect_nodes_set.end()) {
        res += NumAbjacentElementwiseOps(op, {n});
      }
    }
    for (auto* op : n->outputs) {
      if (IsElementwiseOp(op) &&
          expect_nodes_set.find(op) == expect_nodes_set.end()) {
        res += NumAbjacentElementwiseOps(op, {n});
      }
    }
  }
  // LOG(INFO) << "Leave NumAbjacentElementwiseOps: " << n->Name() << ", res="
  // << res;
  return res;
}

void ElementwiseGroupPattern::operator()(PDNode* x, int num_operations) {
  // x->assert_more(IsInputOfElementwiseOp);
  // if (num_operations == 2) {
  //   for (int i = 0; i < num_operations; ++i) {
  //     auto* op = pattern->NewNode(IsElementwiseOp,
  //                                 name_scope_ + "/op_" +std::to_string(i));
  //     auto* out = pattern->NewNode(IsOutputOfElementwiseOp,
  //                                 name_scope_ + "/out_" +std::to_string(i));
  //     ops.push_back(op);
  //     outputs.push_back(out);
  //     if (i == 0) {
  //       ops[i]->LinksFrom({x}).LinksTo({outputs[i]});
  //     } else {
  //       ops[i]->LinksFrom({outputs[i - 1]}).LinksTo({outputs[i]});
  //     }
  //   }
  // }
  x->assert_more(IsInputOfElementwiseOp);
}

}  // namespace patterns
}  // namespace ir
}  // namespace framework
}  // namespace paddle
