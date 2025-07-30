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

#include "paddle/fluid/framework/ir/fusion_group/operation.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"

namespace paddle::framework::ir::fusion_group {

static std::unordered_set<std::string> elementwise_op_types;

static std::unordered_set<std::string>& GetElementwiseOpTypes() {
  if (elementwise_op_types.empty()) {
    elementwise_op_types = OperationMap::Instance().Find(/* type= */ 0);
  }
  return elementwise_op_types;
}

static bool IsSpecifiedOp(const std::unordered_set<std::string>& op_types,
                          const Node* n) {
  if (n && n->IsOp() && n->Op() && !n->outputs.empty()) {
    auto iter = op_types.find(n->Op()->Type());
    if (iter != op_types.end()) {
      return true;
    }
  }
  return false;
}

static bool IsGradOp(const Node* n) {
  PADDLE_ENFORCE_EQ(n && n->IsOp() && n->Op(),
                    true,
                    common::errors::InvalidArgument(
                        "Expected node %p to be an operator node.", n));
  std::string suffix = "_grad";
  std::string op_type = n->Op()->Type();
  size_t pos = op_type.rfind(suffix);
  return pos != std::string::npos &&
         pos == (op_type.length() - suffix.length());
}

static bool IsEqualAndNotEmpty(const std::vector<int64_t>& l,
                               const std::vector<int64_t>& r) {
  return !l.empty() && !r.empty() && l == r;
}

bool GroupDetector::CheckPrecondition(const Node* n) {
  auto check_data_type = [&](const std::vector<Node*>& nodes) -> bool {
    bool is_first = true;
    proto::VarType::Type data_type_0 = proto::VarType::BOOL;
    for (auto* n : nodes) {
      if (n && n->IsVar() && n->Var()) {
        if (n->Var()->GetType() != proto::VarType::DENSE_TENSOR) {
          return false;
        }

        proto::VarType::Type data_type_i = n->Var()->GetDataType();
        if (data_type_i == proto::VarType::FP32 ||
            data_type_i == proto::VarType::FP64 ||
            data_type_i == proto::VarType::FP16) {
          if (is_first) {
            data_type_0 = data_type_i;
            is_first = false;
          } else if (data_type_0 != data_type_i) {
            return false;
          }
        } else {
          return false;
        }
      }
    }
    return true;
  };

  auto check_running_on_cpu = [&](const Node* n) -> bool {
    if (n && n->IsOp() && n->Op()) {
      auto* op = n->Op();
      bool is_run_on_cpu = false;
      if (op->HasAttr("force_cpu") &&
          op->GetAttrType("force_cpu") == proto::AttrType::BOOLEAN) {
        is_run_on_cpu = op->GetAttrIfExists<bool>("force_cpu");
      }
      if (op->HasAttr("op_device")) {
        is_run_on_cpu = op->GetAttrIfExists<std::string>("op_device") == "cpu";
      }
      return is_run_on_cpu;
    }
    return false;
  };

  return n && n->IsOp() && n->Op() && !check_running_on_cpu(n) &&
         check_data_type(n->inputs) && check_data_type(n->outputs);
}

bool ElementwiseGroupDetector::IsElementwiseOp(const Node* n) {
  if (IsSpecifiedOp(GetElementwiseOpTypes(), n)) {
    // Check whether all inputs have the same shape.
    bool is_first = true;
    std::vector<int64_t> shape_0;
    for (auto* in_i : n->inputs) {
      if (in_i && in_i->IsVar() && in_i->Var()) {
        std::vector<int64_t> shape_i = in_i->Var()->GetShape();
        if (is_first) {
          shape_0 = shape_i;
          is_first = false;
        } else {
          if (!IsEqualAndNotEmpty(shape_0, shape_i)) {
            return false;
          }
        }
      }
    }
    auto op = n->Op();
    std::vector<std::string> output_names =
        OperationMap::Instance().Get(op->Type()).output_names;
    for (auto& name : output_names) {
      if (op->Output(name).empty()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::vector<std::vector<Node*>> ElementwiseGroupDetector::operator()(
    Graph* graph) {
  auto teller = [&](const Node* n) -> bool {
    return CheckPrecondition(n) && IsElementwiseOp(n);
  };

  return SubgraphDetector(graph, teller)();
}

}  // namespace paddle::framework::ir::fusion_group
