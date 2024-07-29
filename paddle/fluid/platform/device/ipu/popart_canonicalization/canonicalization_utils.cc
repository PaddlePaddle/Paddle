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

namespace paddle {
namespace platform {
namespace ipu {

std::unordered_map<std::string, SymbolHandler> &SymbolHandlers() {
  static std::unordered_map<std::string, SymbolHandler> symbol_handlers;
  return symbol_handlers;
}

bool RegisterHandler(const std::string &symbol, const SymbolHandler &handler) {
  if (SymbolHandlers().count(symbol) != 0) {
    LOG(WARNING) << "Trying to register popart handler twice for operator: "
                 << symbol;
    return false;
  }
  bool new_handler = SymbolHandlers().emplace(symbol, handler).second;
  return new_handler;
}

SymbolHandler GetHandler(const std::string &kind) {
  auto it = SymbolHandlers().find(kind);
  if (it != SymbolHandlers().end()) {
    return it->second;
  }
  return {};
}

void ConnectNodes(Node *first_node, Node *next_node) {
  first_node->outputs.push_back(next_node);
  next_node->inputs.push_back(first_node);
}

void DisConnectNodes(Node *first_node, Node *next_node) {
  auto rm_by_value = [&](std::vector<Node *> &vec, Node *n) {
    vec.erase(std::remove(vec.begin(), vec.end(), n), vec.end());
  };
  rm_by_value(first_node->outputs, next_node);
  rm_by_value(next_node->inputs, first_node);
  rm_by_value(first_node->inputs, next_node);
  rm_by_value(next_node->outputs, first_node);
}

void ClearNode(Node *node) {
  auto rm_by_value = [&](std::vector<Node *> &vec, Node *n) {
    vec.erase(std::remove(vec.begin(), vec.end(), n), vec.end());
  };
  for (auto *node_in : node->inputs) {
    rm_by_value(node_in->outputs, node);
  }
  for (auto *node_out : node->outputs) {
    rm_by_value(node_out->inputs, node);
  }
}

void CopyOpAttr(const std::string &attr_name,
                OpDesc *op,
                OpDesc *new_op,
                bool override) {
  if (new_op->HasAttr(attr_name) && !override) {
    return;
  }
  if (op->HasAttr(attr_name)) {
    VLOG(10) << "Copying attr: " << attr_name << " from " << op->Type()
             << " to " << new_op->Type();
    new_op->SetAttr(attr_name, op->GetAttr(attr_name));
    new_op->Flush();
  }
}

Node *GetInputVarNode(const std::string &input_name,
                      const Node *op_node,
                      const int id) {
  auto var_name = op_node->Op()->Input(input_name).at(id);
  return GetInputVarNodeByVarName(var_name, op_node);
}

Node *GetOutputVarNode(const std::string &output_name,
                       const Node *op_node,
                       const int id) {
  auto var_name = op_node->Op()->Output(output_name).at(id);
  return GetOutputVarNodeByVarName(var_name, op_node);
}

Node *GetInputVarNodeByVarName(const std::string &var_name,
                               const Node *op_node) {
  for (auto *var : op_node->inputs) {
    if (var->Name() == var_name) {
      return var;
    }
  }
  return nullptr;
}

Node *GetOutputVarNodeByVarName(const std::string &var_name,
                                const Node *op_node) {
  for (auto *var : op_node->outputs) {
    if (var->Name() == var_name) {
      return var;
    }
  }
  return nullptr;
}

const bool is_float_equal(float a, float b, float eps) {
  return std::fabs(a - b) <= eps;
}

const ONNXDataType GetVarDType(const Node *node) {
  auto var = node->Var();
  PADDLE_ENFORCE_NOT_NULL(
      var, common::errors::Unavailable("Node is not a variable."));
  auto proto_var_type = var->GetDataType();
  return VarType2OnnxDType(proto_var_type);
}

const ONNXDataType GetOutputVarDType(const Node *node,
                                     const std::string &output_name) {
  auto out_node = GetOutputVarNode(output_name, node);
  PADDLE_ENFORCE_NOT_NULL(
      out_node, common::errors::Unavailable("Node's out node does not exist."));
  return GetVarDType(out_node);
}

bool IsLastVarNode(Node *node) {
  return node->IsVar() && node->outputs.size() == 0;
}

void MarkNodeForDeletion(Node *node) { node->Op()->SetAttr("delete_node", 1); }

bool IsMarkedForDeletion(Node *node) {
  return node->Op()->HasAttr("delete_node") &&
         PADDLE_GET_CONST(int, node->Op()->GetAttr("delete_node")) > 0;
}

int RemoveTailReduction(Graph *graph,
                        Node *loss_op,
                        const std::string &output_var_name) {
  // Sum: 0. Mean: 1. None: 2
  int reduction = 2;
  Node *reduction_op;
  auto loss_output = GetOutputVarNode(output_var_name, loss_op);
  for (auto sub_node : loss_output->outputs) {
    if (!sub_node->IsOp()) continue;
    if (sub_node->Op()->Type() == "reduce_sum") {
      reduction = 0;
      reduction_op = sub_node;
    } else if (sub_node->Op()->Type() == "reduce_mean") {
      reduction = 1;
      reduction_op = sub_node;
    }
  }
  if (reduction == 2) return reduction;
  auto reduction_out = reduction_op->outputs[0];
  loss_op->Op()->SetOutput(output_var_name,
                           std::vector<std::string>({reduction_out->Name()}));
  MarkNodeForDeletion(reduction_op);
  DisConnectNodes(loss_output, reduction_op);
  DisConnectNodes(reduction_op, reduction_out);
  ConnectNodes(loss_op, reduction_out);

  return reduction;
}

int ConvertToPopartReduction(const std::string &reduction) {
  // Sum: 0. Mean: 1. None: 2
  if (reduction == "sum") {
    return 0;
  } else if (reduction == "mean") {
    return 1;
  } else if (reduction == "none") {
    return 2;
  }
  PADDLE_THROW(common::errors::InvalidArgument(
      "reduction %s is not supported on ipu.", reduction));
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
