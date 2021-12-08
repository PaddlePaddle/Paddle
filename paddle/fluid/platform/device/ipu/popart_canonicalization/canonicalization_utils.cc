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

// This avoids the static initialisation order fiasco,
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

// Return a pointer to a handler if one is registered for this kind of node or
// an empty std::function otherwise.
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

void CopyOpAttr(const std::string &attr_name, OpDesc *op, OpDesc *new_op,
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

const int VarType2OnnxDtype(const int type) {
  auto dtype = static_cast<framework::proto::VarType::Type>(type);
  switch (dtype) {
    case framework::proto::VarType::BOOL:
      return static_cast<int>(ONNXDataType::BOOL);
    case framework::proto::VarType::INT16:
      return static_cast<int>(ONNXDataType::INT16);
    case framework::proto::VarType::INT32:
      return static_cast<int>(ONNXDataType::INT32);
    case framework::proto::VarType::INT64:
      return static_cast<int>(ONNXDataType::INT64);
    case framework::proto::VarType::FP16:
      return static_cast<int>(ONNXDataType::FLOAT16);
    case framework::proto::VarType::FP32:
      return static_cast<int>(ONNXDataType::FLOAT);
    case framework::proto::VarType::FP64:
      return static_cast<int>(ONNXDataType::DOUBLE);
    case framework::proto::VarType::UINT8:
      return static_cast<int>(ONNXDataType::UINT8);
    case framework::proto::VarType::INT8:
      return static_cast<int>(ONNXDataType::INT8);
    case framework::proto::VarType::BF16:
      return static_cast<int>(ONNXDataType::BFLOAT16);
    case framework::proto::VarType::COMPLEX64:
      return static_cast<int>(ONNXDataType::COMPLEX64);
    case framework::proto::VarType::COMPLEX128:
      return static_cast<int>(ONNXDataType::COMPLEX128);
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported data type: %d.", dtype));
  }
}

const std::string VarType2PopStr(const int type) {
  auto dtype = static_cast<framework::proto::VarType::Type>(type);
  switch (dtype) {
    case framework::proto::VarType::UINT8:
      return "UINT8";
    case framework::proto::VarType::INT8:
      return "INT8";
    case framework::proto::VarType::INT16:
      return "INT16";
    case framework::proto::VarType::INT32:
      return "INT32";
    case framework::proto::VarType::INT64:
      return "INT64";
    case framework::proto::VarType::BOOL:
      return "BOOL";
    case framework::proto::VarType::FP64:
      return "DOUBLE";
    case framework::proto::VarType::FP32:
      return "FLOAT";
    case framework::proto::VarType::FP16:
      return "FLOAT16";
    default:
      PADDLE_THROW(
          paddle::platform::errors::Unavailable("Unsupported data type."));
  }
}

Node *GetInputVarNode(const std::string &input_name, const Node *op_node,
                      const int id) {
  auto var_name = op_node->Op()->Input(input_name).at(id);
  return GetInputVarNodeByVarName(var_name, op_node);
}

Node *GetOutputVarNode(const std::string &output_name, const Node *op_node,
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

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
