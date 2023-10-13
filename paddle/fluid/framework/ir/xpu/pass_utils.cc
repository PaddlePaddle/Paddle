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

static void HashCombine(std::size_t* seed) {}

// combine hash value
// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
template <typename T, typename... Rest>
static void HashCombine(std::size_t* seed, const T& v, Rest... rest) {
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
  *seed *= 0x00000100000001B3;
  HashCombine(seed, rest...);
}

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
  } else if (act_type == "elu") {
    return static_cast<int>(xpu::Activation_t::ELU);
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

std::vector<Node*> FindOpNodeByInputName(Graph* graph,
                                         const std::string& var_name) {
  std::vector<Node*> ret;
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) continue;
    auto inputs = node->Op()->Inputs();
    for (auto input : inputs) {
      auto in_names = input.second;
      if (std::count(in_names.begin(), in_names.end(), var_name) > 0) {
        ret.push_back(node);
        break;
      }
    }
  }
  return ret;
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
size_t HashTensor(const phi::DenseTensor& in) {
  size_t ret = 0;
  auto in_dims = in.dims();
  HashCombine(&ret,
              phi::DataTypeToString(in.dtype()),
              phi::DataLayoutToString(in.layout()),
              in_dims.size());
  for (int i = 0; i < in_dims.size(); i++) {
    HashCombine(&ret, in_dims[i]);
  }

  auto* data = in.data<T>();
  int64_t size = in.numel();
  for (int64_t i = 0; i < size; i++) {
    HashCombine(&ret, data[i]);
  }
  return ret;
}

template size_t HashTensor<int16_t>(const phi::DenseTensor& in);
template size_t HashTensor<float>(const phi::DenseTensor& in);

std::string GetPrefixWithoutHash(const std::string& name) {
  std::size_t found = name.find("_#");
  return found == std::string::npos ? name : name.substr(0, found);
}

template <typename T>
void PrepareWeight(Graph* graph,
                   Scope* scope,
                   BlockDesc* block,
                   Node* src,
                   Node** dst,
                   Node** dst_max,
                   bool transpose) {
  auto src_name = src->Name();
  auto* src_tensor = scope->Var(src_name)->GetMutable<phi::DenseTensor>();
  phi::DenseTensor dst_tensor;
  Assign(*src_tensor, &dst_tensor);
  phi::DenseTensor dst_max_tensor;
  PrepareWeight<T>(&dst_tensor, &dst_max_tensor, transpose);

  size_t dst_hash = HashTensor<T>(dst_tensor);
  size_t dst_max_hash = HashTensor<float>(dst_max_tensor);
  std::string pre_name = GetPrefixWithoutHash(src_name);
  std::string dst_name = pre_name + "_#" + std::to_string(dst_hash);
  std::string dst_max_name = pre_name + "_max_#" + std::to_string(dst_max_hash);
  *dst = FindNodeWithName(graph, dst_name);
  if (*dst == nullptr) {
    // Create dst node
    // Update dst var_desc in block
    VarDesc dst_desc(dst_name);
    dst_desc.SetPersistable(true);
    dst_desc.SetShape(vectorize(dst_tensor.dims()));
    dst_desc.SetDataType(framework::TransToProtoVarType(dst_tensor.dtype()));
    *dst = graph->CreateVarNode(&dst_desc);
    auto* block_dst_desc = block->Var(dst_name);
    block_dst_desc->SetPersistable(dst_desc.Persistable());
    block_dst_desc->SetShape(dst_desc.GetShape());
    block_dst_desc->SetDataType(dst_desc.GetDataType());
    // Create dst_max node
    // Update dst_max var_desc in block
    VarDesc dst_max_desc(dst_max_name);
    dst_max_desc.SetPersistable(true);
    dst_max_desc.SetShape(vectorize(dst_max_tensor.dims()));
    dst_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    *dst_max = graph->CreateVarNode(&dst_max_desc);
    auto* block_dst_max_desc = block->Var(dst_max_name);
    block_dst_max_desc->SetPersistable(dst_max_desc.Persistable());
    block_dst_max_desc->SetShape(dst_max_desc.GetShape());
    block_dst_max_desc->SetDataType(dst_max_desc.GetDataType());

    // Find dst/dst_max variable in scope
    auto* dst_var = scope->FindVar(dst_name);
    if (dst_var == nullptr) {
      // Create dst/dst_max variable/tensor
      Assign(dst_tensor, scope->Var(dst_name)->GetMutable<phi::DenseTensor>());
      Assign(dst_max_tensor,
             scope->Var(dst_max_name)->GetMutable<phi::DenseTensor>());
    } else {
      // Share the same variable
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(dst_max_name),
          platform::errors::Fatal(
              "dst_max(%s) variable should not be nullptr if dst(%s) "
              "variable is exist. (src_name is %s)",
              dst_max_name,
              dst_name,
              src_name));
    }
  } else {
    *dst_max = FindNodeWithName(graph, dst_max_name);
    PADDLE_ENFORCE_NOT_NULL(
        *dst_max,
        platform::errors::Fatal(
            "dst_max(%s) variable should not be nullptr if dst(%s) "
            "variable is exist. (src_name is %s)",
            dst_max_name,
            dst_name,
            src_name));
  }
}

template void PrepareWeight<int16_t>(Graph* graph,
                                     Scope* scope,
                                     BlockDesc* block,
                                     Node* src,
                                     Node** dst,
                                     Node** dst_max,
                                     bool transpose);
template void PrepareWeight<int8_t>(Graph* graph,
                                    Scope* scope,
                                    BlockDesc* block,
                                    Node* src,
                                    Node** dst,
                                    Node** dst_max,
                                    bool transpose);

void PrepareBias(
    Graph* graph, Scope* scope, BlockDesc* block, Node* src, Node** dst) {
  auto src_name = src->Name();
  auto* src_tensor = scope->Var(src_name)->GetMutable<phi::DenseTensor>();
  if (src_tensor->dtype() == phi::DataType::FLOAT32) {
    *dst = src;
  }

  phi::DenseTensor dst_tensor;
  CastToFp32(src_tensor, &dst_tensor);
  size_t dst_hash = HashTensor<float>(dst_tensor);
  std::string pre_name = GetPrefixWithoutHash(src_name);
  std::string dst_name = pre_name + "_#" + std::to_string(dst_hash);
  *dst = FindNodeWithName(graph, dst_name);
  if (*dst == nullptr) {
    // Create dst node
    // Update dst var_desc in block
    VarDesc dst_desc(dst_name);
    dst_desc.SetPersistable(true);
    dst_desc.SetShape(vectorize(dst_tensor.dims()));
    dst_desc.SetDataType(framework::TransToProtoVarType(dst_tensor.dtype()));
    *dst = graph->CreateVarNode(&dst_desc);
    auto* block_dst_desc = block->Var(dst_name);
    block_dst_desc->SetPersistable(dst_desc.Persistable());
    block_dst_desc->SetShape(dst_desc.GetShape());
    block_dst_desc->SetDataType(dst_desc.GetDataType());
    Assign(dst_tensor, scope->Var(dst_name)->GetMutable<phi::DenseTensor>());
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
