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
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/fusion_group/operation.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

static std::unordered_set<std::string> binary_op_types;
static std::unordered_set<std::string> unary_op_types;

static std::unordered_set<std::string>& GetBinaryOpTypes() {
  if (binary_op_types.empty()) {
    binary_op_types =
        OperationMap::Instance().Find(/* type= */ 0, /* num_operands= */ 2);
  }
  return binary_op_types;
}

static std::unordered_set<std::string>& GetUnaryOpTypes() {
  if (unary_op_types.empty()) {
    unary_op_types =
        OperationMap::Instance().Find(/* type= */ 0, /* num_operands= */ 1);
  }
  return unary_op_types;
}

static bool IsSpecifiedOp(const std::unordered_set<std::string>& op_types,
                          const Node* n) {
  if (n && n->IsOp() && n->Op() && n->outputs.size() > 0U) {
    auto iter = op_types.find(n->Op()->Type());
    if (iter != op_types.end()) {
      return true;
    }
  }
  return false;
}

static bool IsGradOp(const Node* n) {
  PADDLE_ENFORCE_EQ(n && n->IsOp() && n->Op(), true,
                    platform::errors::InvalidArgument(
                        "Expected node %p to be an operator node.", n));
  std::string suffix = "_grad";
  std::string op_type = n->Op()->Type();
  size_t pos = op_type.rfind(suffix);
  return pos != std::string::npos &&
         pos == (op_type.length() - suffix.length());
}

static bool IsEqualAndNotEmpty(const std::vector<int64_t>& l,
                               const std::vector<int64_t>& r) {
  return l.size() != 0U && r.size() != 0U && l == r;
}

static bool IsBinaryOp(const Node* n) {
  if (IsSpecifiedOp(GetBinaryOpTypes(), n)) {
    if ((!IsGradOp(n) && n->inputs.size() != 2U) || n->inputs.size() == 0U) {
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
        if (!IsEqualAndNotEmpty(shape_0, shape_i)) {
          return false;
        }
      }
    }
    return true;
  }
  return false;
}

static bool IsUnaryOp(const Node* n) {
  return IsSpecifiedOp(GetUnaryOpTypes(), n);
}

bool ElementwiseGroupDetector::IsElementwiseOp(const Node* n) {
  return IsBinaryOp(n) || IsUnaryOp(n);
}

std::vector<std::vector<Node*>> ElementwiseGroupDetector::operator()(
    Graph* graph) {
  auto teller = [&](const Node* n) -> bool { return IsElementwiseOp(n); };

  return SubgraphDetector(graph, teller)();
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
