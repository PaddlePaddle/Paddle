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
#include "paddle/phi/kernels/cast_kernel.h"
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
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Not support convert activation_type(%s).", act_type));
  }
  return -1;
}

Node* FindNodeWithName(Graph* graph, std::string name) {
  for (auto* node : graph->Nodes()) {
    if (node->IsVar() && node->Var() && node->Var()->Name() == name) {
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
  PADDLE_THROW(common::errors::InvalidArgument("Not support type."));
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
              common::DataLayoutToString(in.layout()),
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
template size_t HashTensor<int8_t>(const phi::DenseTensor& in);

template <>
size_t HashTensor<float16>(const phi::DenseTensor& in) {
  phi::DenseTensor dst_tensor;
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  dst_tensor.Resize(in.dims());
  dst_tensor.set_type(phi::DataType::FLOAT32);
  dst_tensor.set_layout(in.layout());
  phi::CastKernel<float16>(*cpu_ctx, in, phi::DataType::FLOAT32, &dst_tensor);
  return HashTensor<float>(dst_tensor);
}

std::string GetPrefixWithoutHash(const std::string& name) {
  std::size_t found = name.find("_#");
  return found == std::string::npos ? name : name.substr(0, found);
}

void ConvertFromFp32ToFp16(phi::DenseTensor* weight,
                           phi::DenseTensor* weight_max,
                           bool transpose) {
  // Convert fp16 to fp32
  phi::DenseTensor weight_fp32;
  CastToFp32(weight, &weight_fp32);

  if (transpose) {  // (k, n) -> (n, k)
    Transpose2D(&weight_fp32);
  }

  auto FindMaxAbs = [](const float* data, int len) {
    float max_f = 0.0f;
    for (int i = 0; i < len; ++i) {
      float max = std::abs(data[i]);
      if (max > max_f) {
        max_f = max;
      }
    }
    return max_f;
  };

  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  // Convert to fp16
  phi::DenseTensor weight_fp16;
  CastToFp16(&weight_fp32, &weight_fp16);
  // Find max
  int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
  int size = weight_fp32.numel();
  float max_val = FindMaxAbs(weight_fp32.data<float>(), size);
  std::vector<float> max_vec(max_ptr_size, max_val);
  weight_max->set_type(phi::DataType::FLOAT32);
  weight_max->Resize({max_ptr_size});
  memcpy(cpu_ctx->Alloc<float>(weight_max),
         max_vec.data(),
         max_ptr_size * sizeof(float));
  weight->clear();
  weight->set_type(phi::DataType::FLOAT16);
  weight->Resize({size});
  memcpy(cpu_ctx->Alloc<float16>(weight),
         weight_fp16.data<float16>(),
         size * sizeof(float16));
}

template <typename Tcpu, typename Txpu>
void PrepareWeight(Graph* graph,
                   Scope* scope,
                   BlockDesc* block,
                   Node* weight,
                   Node** dst_weight,
                   Node** dst_weight_max,
                   Node** dst_scale_max,
                   bool transpose,
                   const std::vector<float>& weight_scales,
                   bool per_channel_quant) {
  auto weight_name = weight->Name();
  auto* weight_tensor = scope->Var(weight_name)->GetMutable<phi::DenseTensor>();
  phi::DenseTensor dst_weight_tensor;
  Assign(*weight_tensor, &dst_weight_tensor);
  phi::DenseTensor dst_weight_max_tensor;
  phi::DenseTensor dst_scale_max_tensor;
  ConvertWeightWrapper<Tcpu, Txpu>(&dst_weight_tensor,
                                   &dst_weight_max_tensor,
                                   &dst_scale_max_tensor,
                                   transpose,
                                   weight_scales,
                                   per_channel_quant);

  size_t dst_weight_hash = HashTensor<Txpu>(dst_weight_tensor);
  size_t dst_weight_max_hash = HashTensor<float>(dst_weight_max_tensor);
  std::string pre_name = GetPrefixWithoutHash(weight_name);
  std::string dst_weight_name =
      pre_name + "_#" + std::to_string(dst_weight_hash);
  std::string dst_weight_max_name =
      pre_name + "_max_#" + std::to_string(dst_weight_max_hash);

  *dst_weight = FindNodeWithName(graph, dst_weight_name);
  if (*dst_weight == nullptr) {
    // Create dst_weight node
    // Update dst_weight var_desc in block
    VarDesc dst_weight_desc(dst_weight_name);
    dst_weight_desc.SetPersistable(true);
    dst_weight_desc.SetShape(common::vectorize(dst_weight_tensor.dims()));
    dst_weight_desc.SetDataType(
        framework::TransToProtoVarType(dst_weight_tensor.dtype()));
    *dst_weight = graph->CreateVarNode(&dst_weight_desc);
    auto* block_dst_weight_desc = block->Var(dst_weight_name);
    block_dst_weight_desc->SetPersistable(dst_weight_desc.Persistable());
    block_dst_weight_desc->SetShape(dst_weight_desc.GetShape());
    block_dst_weight_desc->SetDataType(dst_weight_desc.GetDataType());
    // Create dst_weight_max node
    // Update dst_weight_max var_desc in block
    VarDesc dst_weight_max_desc(dst_weight_max_name);
    dst_weight_max_desc.SetPersistable(true);
    dst_weight_max_desc.SetShape(
        common::vectorize(dst_weight_max_tensor.dims()));
    dst_weight_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
    *dst_weight_max = graph->CreateVarNode(&dst_weight_max_desc);
    auto* block_dst_weight_max_desc = block->Var(dst_weight_max_name);
    block_dst_weight_max_desc->SetPersistable(
        dst_weight_max_desc.Persistable());
    block_dst_weight_max_desc->SetShape(dst_weight_max_desc.GetShape());
    block_dst_weight_max_desc->SetDataType(dst_weight_max_desc.GetDataType());
    // Find dst/dst_max variable in scope
    auto* dst_weight_var = scope->FindVar(dst_weight_name);
    if (dst_weight_var == nullptr) {
      // Create dst_weight/dst_weight_max variable/tensor
      Assign(dst_weight_tensor,
             scope->Var(dst_weight_name)->GetMutable<phi::DenseTensor>());
      Assign(dst_weight_max_tensor,
             scope->Var(dst_weight_max_name)->GetMutable<phi::DenseTensor>());
    } else {
      // Share the same variable
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(dst_weight_max_name),
          common::errors::Fatal("dst_weight_max(%s) variable should not be "
                                "nullptr if dst_weight(%s) "
                                "variable is exist. (weight_name is %s)",
                                dst_weight_max_name,
                                dst_weight_name,
                                weight_name));
    }
  } else {
    *dst_weight_max = FindNodeWithName(graph, dst_weight_max_name);
    PADDLE_ENFORCE_NOT_NULL(
        *dst_weight_max,
        common::errors::Fatal("dst_weight_max(%s) variable should not be "
                              "nullptr if dst_weight(%s) "
                              "variable is exist. (weight_name is %s)",
                              dst_weight_max_name,
                              dst_weight_name,
                              weight_name));
  }

  if (dst_scale_max_tensor.initialized()) {
    size_t dst_scale_max_hash = HashTensor<float>(dst_scale_max_tensor);
    std::string dst_scale_max_name =
        pre_name + "_scale_max_#" + std::to_string(dst_scale_max_hash);
    if (*dst_scale_max == nullptr) {
      // Create dst_scale_max node
      // Update dst_scale_max var_desc in block
      VarDesc dst_scale_max_desc(dst_scale_max_name);
      dst_scale_max_desc.SetPersistable(true);
      dst_scale_max_desc.SetShape(
          common::vectorize(dst_weight_max_tensor.dims()));
      dst_scale_max_desc.SetDataType(proto::VarType::Type::VarType_Type_FP32);
      *dst_scale_max = graph->CreateVarNode(&dst_scale_max_desc);
      auto* block_dst_scale_max_desc = block->Var(dst_scale_max_name);
      block_dst_scale_max_desc->SetPersistable(
          dst_scale_max_desc.Persistable());
      block_dst_scale_max_desc->SetShape(dst_scale_max_desc.GetShape());
      block_dst_scale_max_desc->SetDataType(dst_scale_max_desc.GetDataType());
      // Find dst/dst_max variable in scope
      auto* dst_scale_max_var = scope->FindVar(dst_scale_max_name);
      if (dst_scale_max_var == nullptr) {
        Assign(dst_scale_max_tensor,
               scope->Var(dst_scale_max_name)->GetMutable<phi::DenseTensor>());
      } else {
        // Share the same variable
        PADDLE_ENFORCE_NOT_NULL(
            scope->FindVar(dst_scale_max_name),
            common::errors::Fatal("dst_scale_max(%s) variable should not be "
                                  "nullptr if dst_weight(%s) "
                                  "variable is exist. (weight_name is %s)",
                                  dst_scale_max_name,
                                  dst_weight_name,
                                  weight_name));
      }
    }
  }
}

template void PrepareWeight<float, float>(
    Graph* graph,
    Scope* scope,
    BlockDesc* block,
    Node* weight,
    Node** dst_weight,
    Node** dst_weight_max,
    Node** dst_scale_max,
    bool transpose,
    const std::vector<float>& weight_scales,
    bool per_channel_quant = false);

template void PrepareWeight<float, float16>(
    Graph* graph,
    Scope* scope,
    BlockDesc* block,
    Node* weight,
    Node** dst_weight,
    Node** dst_weight_max,
    Node** dst_scale_max,
    bool transpose,
    const std::vector<float>& weight_scales,
    bool per_channel_quant = false);

template void PrepareWeight<float, int16_t>(
    Graph* graph,
    Scope* scope,
    BlockDesc* block,
    Node* weight,
    Node** dst_weight,
    Node** dst_weight_max,
    Node** dst_scale_max,
    bool transpose,
    const std::vector<float>& weight_scales,
    bool per_channel_quant = false);

template void PrepareWeight<float, int8_t>(
    Graph* graph,
    Scope* scope,
    BlockDesc* block,
    Node* weight,
    Node** dst_weight,
    Node** dst_weight_max,
    Node** dst_scale_max,
    bool transpose,
    const std::vector<float>& weight_scales,
    bool per_channel_quant = false);

template void PrepareWeight<int8_t, int8_t>(
    Graph* graph,
    Scope* scope,
    BlockDesc* block,
    Node* weight,
    Node** dst_weight,
    Node** dst_weight_max,
    Node** dst_scale_max,
    bool transpose,
    const std::vector<float>& weight_scales,
    bool per_channel_quant = false);

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
    dst_desc.SetShape(common::vectorize(dst_tensor.dims()));
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
