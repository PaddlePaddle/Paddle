// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace paddle {
namespace framework {
namespace ir {

int ConvertActivationType(std::string act_type) {
  if (act_type == "") {
    return static_cast<int>(xpu::Activation_t::LINEAR);
  } else if (act_type == "relu") {
    return static_cast<int>(xpu::Activation_t::RELU);
  } else if (act_type == "sigmoid") {
    return static_cast<int>(xpu::Activation_t::SIGMOID);
  } else if (act_type == "tanh") {
    return static_cast<int>(xpu::Activation_t::TANH);
  } else if (act_type == "gelu") {
    return static_cast<int>(xpu::Activation_t::GELU);
  } else if (act_type == "leaky_relu") {
    return static_cast<int>(xpu::Activation_t::LEAKY_RELU);
  } else if (act_type == "exp") {
    return static_cast<int>(xpu::Activation_t::EXP);
  } else if (act_type == "hard_swish") {
    return static_cast<int>(xpu::Activation_t::HARD_SWISH);
  } else if (act_type == "hard_sigmoid") {
    return static_cast<int>(xpu::Activation_t::HARD_SIGMOID);
  } else if (act_type == "swish") {
    return static_cast<int>(xpu::Activation_t::SWISH);
  } else if (act_type == "relu6") {
    return static_cast<int>(xpu::Activation_t::RELU6);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Not support convert activation_type(%s).", act_type));
  }
  return -1;
}

Node* FindNodeWithName(Graph* graph, std::string name) {
  for (auto* node : graph->Nodes()) {
    if (node->IsVar() && node->Var()->Name() == name) {
      return node;
    }
  }
  return nullptr;
}

template <typename T>
std::string IntTypeToString() {
  LOG(FATAL) << "Not support type.";
  return "";
}

template <>
std::string IntTypeToString<int16_t>() {
  return "int16";
}

template <typename T>
void PrepareWeight(Graph* graph,
                   Scope* scope,
                   BlockDesc* block,
                   Node* src_w,
                   Node** dst_w,
                   Node** dst_w_max,
                   bool transpose) {
  auto src_w_name = src_w->Name();
  std::string dst_w_name = src_w_name + "_" + IntTypeToString<T>();
  *dst_w = FindNodeWithName(graph, dst_w_name);
  std::string dst_w_max_name = src_w_name + "_max";
  *dst_w_max = nullptr;

  if (*dst_w == nullptr) {
    *dst_w_max = FindNodeWithName(graph, dst_w_max_name);
    PADDLE_ENFORCE(
        *dst_w_max == nullptr,
        platform::errors::Fatal(
            "dst_w_max(%s) node should be nullptr if dst_w node is nullptr.",
            dst_w_max_name));
    // Create dst_w node
    // Update dst_w var_desc in block
    auto* src_w_desc = src_w->Var();
    VarDesc dst_w_desc(dst_w_name);
    dst_w_desc.SetPersistable(src_w_desc->Persistable());
    dst_w_desc.SetShape(src_w_desc->GetShape());
    dst_w_desc.SetDataType(src_w_desc->GetDataType());
    *dst_w = graph->CreateVarNode(&dst_w_desc);
    auto* block_dst_w_desc = block->Var(dst_w_name);
    block_dst_w_desc->SetPersistable(src_w_desc->Persistable());
    block_dst_w_desc->SetShape(src_w_desc->GetShape());
    block_dst_w_desc->SetDataType(src_w_desc->GetDataType());
    // Create dst_w_max node
    // Update dst_w_max var_desc in block
    VarDesc dst_w_max_desc(dst_w_max_name);
    dst_w_max_desc.SetPersistable(true);
    dst_w_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    *dst_w_max = graph->CreateVarNode(&dst_w_max_desc);
    auto* block_dst_w_max_desc = block->Var(dst_w_max_name);
    block_dst_w_max_desc->SetPersistable(true);
    block_dst_w_max_desc->SetDataType(proto::VarType::Type::VarType_Type_FP32);

    // Find dst_w/dst_w_max variable in scope
    auto* dst_w_var = scope->FindVar(dst_w_name);
    if (dst_w_var == nullptr) {
      PADDLE_ENFORCE(
          scope->FindVar(dst_w_max_name) == nullptr,
          platform::errors::Fatal("dst_w_max(%s) variable should be nullptr if "
                                  "dst_w variable is nullptr.",
                                  dst_w_max_name));
      // Create dst_w/dst_w_max variable/tensor
      auto* src_w_tensor =
          scope->Var(src_w_name)->GetMutable<phi::DenseTensor>();
      auto* dst_w_tensor =
          scope->Var(dst_w_name)->GetMutable<phi::DenseTensor>();
      Assign(*src_w_tensor, dst_w_tensor);
      auto* dst_w_max_tensor =
          scope->Var(dst_w_max_name)->GetMutable<phi::DenseTensor>();
      PrepareWeight<int16_t>(dst_w_tensor, dst_w_max_tensor, transpose);
    } else {
      // Share the same variable
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(dst_w_max_name),
          platform::errors::Fatal("dst_w_max(%s) variable should not be "
                                  "nullptr if dst_w variable is exist.",
                                  dst_w_max_name));
    }
  } else {
    *dst_w_max = FindNodeWithName(graph, dst_w_max_name);
    PADDLE_ENFORCE_NOT_NULL(
        *dst_w_max,
        platform::errors::Fatal(
            "dst_w_max(%s) node should not be nullptr if dst_w node is exist.",
            dst_w_max_name));
  }
}

template void PrepareWeight<int16_t>(Graph* graph,
                                     Scope* scope,
                                     BlockDesc* block,
                                     Node* src_w,
                                     Node** dst_w,
                                     Node** dst_w_max,
                                     bool transpose);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
