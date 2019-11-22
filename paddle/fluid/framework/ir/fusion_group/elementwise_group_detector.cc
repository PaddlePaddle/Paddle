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
#include "paddle/fluid/framework/ir/fusion_group/operation.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

static std::unordered_set<std::string> binary_op_types;
static std::unordered_set<std::string> unary_op_types;

static std::unordered_set<std::string>& GetBinaryOpTypes() {
  if (binary_op_types.empty()) {
    binary_op_types = OperationMap::Instance().Find(0, 2);
  }
  return binary_op_types;
}

static std::unordered_set<std::string>& GetUnaryOpTypes() {
  if (unary_op_types.empty()) {
    unary_op_types = OperationMap::Instance().Find(0, 1);
  }
  return unary_op_types;
}

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

static bool IsGradOp(Node* n) {
  PADDLE_ENFORCE_EQ(n && n->IsOp() && n->Op(), true,
                    "Node %p should be a op node.", n);
  std::string suffix = "_grad";
  std::string op_type = n->Op()->Type();
  size_t pos = op_type.rfind(suffix);
  return pos != std::string::npos &&
         pos == (op_type.length() - suffix.length());
}

static bool IsEqual(const std::vector<int64_t>& l,
                    const std::vector<int64_t>& r) {
  if (l.size() == 0U || r.size() == 0U || l.size() != r.size()) {
    return false;
  }
  for (size_t i = 0; i < l.size(); ++i) {
    if (l[i] != r[i]) {
      return false;
    }
  }
  return true;
}

static bool IsBinaryOp(Node* n, bool backward) {
  if (IsSpecifiedOp(GetBinaryOpTypes(), n) && (IsGradOp(n) == backward)) {
    if ((!backward && n->inputs.size() != 2U) || n->inputs.size() == 0U) {
      return false;
    }

    // The shape of all inputs should be the same.
    std::vector<int64_t> shape_0;
    for (size_t i = 0; i < n->inputs.size(); ++i) {
      auto* in_i = n->inputs[i];
      if (!(in_i && in_i->IsVar() && in_i->Var())) {
        return false;
      }

      std::vector<int64_t> shape_i = in_i->Var()->GetShape();
      if (i == 0U) {
        shape_0 = shape_i;
      } else {
        if (!IsEqual(shape_0, shape_i)) {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

static bool IsUnaryOp(Node* n, bool backward) {
  return IsSpecifiedOp(GetUnaryOpTypes(), n) && (IsGradOp(n) == backward);
}

void ElementwiseGroupDetector::Init(Graph* graph, bool backward) {
  graph_ = graph;
  backward_ = backward;
  for (auto* n : graph_->Nodes()) {
    if (IsBinaryOp(n, backward) || IsUnaryOp(n, backward)) {
      elementwise_ops_.insert(n);
    }
  }
  // LOG(INFO) << "elementise ops for graph:" << graph
  //           << ", backward=" << backward;
  // LOG(INFO) << "{\n" << DebugString(elementwise_ops_) << "}\n";
}

bool ElementwiseGroupDetector::IsElementwiseOp(Node* n) {
  if (n && n->IsOp() && n->Op()) {
    return elementwise_ops_.find(n) != elementwise_ops_.end();
  } else {
    return false;
  }
}

bool ElementwiseGroupDetector::IsInputOfElementwiseOp(Node* n,
                                                      std::string name) {
  if (n && n->IsVar() && n->Var()) {
    for (auto* op : n->outputs) {
      if (IsElementwiseOp(op)) {
        if (name.empty()) {
          return true;
        } else {
          auto var_name = op->Op()->Input(name);
          if (var_name.size() == 1U && var_name[0] == n->Name()) {
            return true;
          }
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

int ElementwiseGroupDetector::Search(Node* n, std::vector<Node*> except_nodes,
                                     SubGraph* subgraph) {
  std::unordered_set<Node*> except_nodes_set;
  for (size_t i = 0; i < except_nodes.size(); ++i) {
    except_nodes_set.insert(except_nodes[i]);
  }

  auto search_op_handler = [&](Node* n, Node* var) -> int {
    // n, is a op node.
    // var, is n's input or output var node.
    int num_operations = 0;
    if (var && var->IsVar() && var->Var() && !subgraph->Has(var)) {
      subgraph->Insert(var);
      if (except_nodes_set.find(var) == except_nodes_set.end()) {
        num_operations = Search(var, {n}, subgraph);
      }
    }
    return num_operations;
  };

  auto search_var_handler = [&](Node* n, Node* op) -> int {
    // n, is a var node.
    // op, is n's input or output op node.
    int num_operations = 0;
    if (IsElementwiseOp(op) &&
        except_nodes_set.find(op) == except_nodes_set.end() &&
        !subgraph->Has(op)) {
      num_operations = Search(op, {n}, subgraph);
    }
    return num_operations;
  };

  int num_operations = 0;
  if (IsElementwiseOp(n)) {
    // LOG(INFO) << "Search[begin]:" << n->Op()->Type();
    subgraph->Insert(n);
    num_operations += 1;
    for (auto* var : n->inputs) {
      num_operations += search_op_handler(n, var);
    }
    for (auto* var : n->outputs) {
      num_operations += search_op_handler(n, var);
    }
  } else if (n && n->IsVar() && n->Var()) {
    // LOG(INFO) << "Search[begin]:" << n->Name();
    for (auto* op : n->inputs) {
      num_operations += search_var_handler(n, op);
    }
    for (auto* op : n->outputs) {
      num_operations += search_var_handler(n, op);
    }
  }
  return num_operations;
}

SubGraph ElementwiseGroupDetector::operator()(Node* n) {
  SubGraph subgraph(0);
  if (!IsOutputOfElementwiseOp(n) && IsInputOfElementwiseOp(n, "X")) {
    LOG(INFO) << "Begin with node:" << n->Name() << ", backward:" << backward_;
    subgraph.Insert(n);
    int num_operations = Search(n, n->inputs, &subgraph);
    VLOG(3) << "Detect elementwise subgraph begin with " << n->Name() << ", "
            << num_operations << " operations, " << subgraph.GetNumNodes()
            << " nodes";
  }
  return subgraph;
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
