// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

Node *topK_op_handler(Graph *graph, Node *node) {
  VLOG(10) << "[topK_op_handler] entering to handler ...";
  auto *op = node->Op();
  auto attrs = AttributeMap{};
  int axis_32INT = -1;
  if (op->HasAttr("axis")) {
    axis_32INT = BOOST_GET_CONST(int, op->GetAttr("axis"));
  }
  if (axis_32INT == -1) {
    auto shape = GetInputVarNode("X", node)->Var()->GetShape();
    int rank = shape.size();
    if (rank < 1) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The dimension of the shape of topK input should be large than 1"));
    }
    axis_32INT = rank - 1;
  }
  int64_t axis = int64_t{axis_32INT};
  attrs.emplace("axis", axis);

  bool largest = true;
  if (op->HasAttr("largest")) {
    largest = BOOST_GET_CONST(bool, op->GetAttr("largest"));
  }
  if (largest) {
    // defaults to 1, largest values
    attrs.emplace("largest", 1);
  } else {
    attrs.emplace("largest", 0);
  }

  bool sorted = true;
  if (op->HasAttr("sorted")) {
    sorted = BOOST_GET_CONST(bool, op->GetAttr("sorted"));
  }
  if (sorted) {
    // defaults to 1, sorted results
    attrs.emplace("sorted", 1);
  } else {
    attrs.emplace("sorted", 0);
  }

  std::vector<paddle::framework::ir::Node *> inputs = node->inputs;
  if (node->inputs.size() == 2) {
    // Input X tensor and K const tensor
    VLOG(10) << "[topK_op_handler] get 2 input tensors.";
    inputs[0] = node->inputs[1];  // K_t
    VLOG(10) << "[topK_op_handler] input node(" << inputs[0]->Var()->Name()
             << ")";
    inputs[1] = node->inputs[0];  // X
    VLOG(10) << "[topK_op_handler] input node(" << inputs[1]->Var()->Name()
             << ")";
  } else if (node->inputs.size() == 1) {
    // Input X tensor with k integer
    VLOG(10) << "[topK_op_handler] get 1 input tensor.";
    int k_32INT = BOOST_GET_CONST(int, op->GetAttr("k"));
    int64_t k = int64_t{k_32INT};
    attrs.emplace("k", k);
  }
  // show output node dtype
  for (auto *o_node : node->outputs) {
    auto *var = o_node->Var();
    // see framework.pb.h
    // VarType_Type_INT64 = 3,
    // VarType_Type_FP32 = 5,
    auto dtype = var->GetDataType();
    if (dtype == 3) {
      // poplar does not support int64_t
      var->SetDataType(framework::proto::VarType::INT32);
    }
    std::string name = var->Name();
    VLOG(10) << "[topK_op_handler] output node(" << name
             << ") dtype : " << dtype;
  }
  VLOG(10) << "[topK_op_handler] leave the handler.";
  return CreateBaseOp(graph, node, "popart_topk", inputs,
                      {node->outputs[1], node->outputs[0]}, attrs);
}

REGISTER_HANDLER(top_k, topK_op_handler);
REGISTER_HANDLER(top_k_v2, topK_op_handler);

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle
