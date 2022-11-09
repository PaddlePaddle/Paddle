// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/float_to_mixed_pass.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace framework {
namespace ir {

namespace {
using VarType = FloatToMixedPass::VarType;

bool PhiKernelSupportPrecision(
    const std::string& op_type,
    phi::Backend backend,
    phi::DataType data_type,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  const auto& kernels = phi::KernelFactory::Instance().kernels();
  if (kernels.count(op_type)) {
    return false;
  }
  phi::KernelKey kernel_key(backend, layout, data_type);
  return phi::KernelFactory::Instance().HasKernel(op_type, kernel_key);
}

bool GpuKernelSupportPrecision(
    const std::string& op_type,
    phi::DataType precision,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  auto phi_op_type = phi::TransToPhiKernelName(op_type);
  bool support = PhiKernelSupportPrecision(
      phi_op_type, phi::Backend::GPU, precision, layout);
  support |= PhiKernelSupportPrecision(
      phi_op_type, phi::Backend::GPUDNN, precision, layout);

  if (!support) {
    const auto& all_kernels = framework::OperatorWithKernel::AllOpKernels();
    auto it = all_kernels.find(op_type);
    if (it != all_kernels.end()) {
      for (const auto& kern_pair : it->second) {
        if (platform::is_gpu_place(kern_pair.first.place_) &&
            kern_pair.first.data_type_ ==
                framework::TransToProtoVarType(precision)) {
          support = true;
          break;
        }
      }
    }
  }
  return support;
}

void DoInsertCastOp(
    framework::ir::Graph* graph,
    framework::ir::Node* var_node,
    framework::ir::Node* op_node,
    VarType::Type from_type,
    VarType::Type to_type,
    framework::BlockDesc* block_desc,
    int* suffix,
    std::unordered_map<framework::ir::Node*, framework::ir::Node*>* cache) {
  if (from_type == to_type) return;

  auto update_cast_desc = [&](framework::OpDesc& desc,
                              const std::string& x_name,
                              const std::string& out_name,
                              const int in_dtype,
                              const int out_dtype) {
    desc.SetType("cast");
    desc.SetInput("X", {x_name});
    desc.SetOutput("Out", {out_name});
    desc.SetAttr("in_dtype", in_dtype);
    desc.SetAttr("out_dtype", out_dtype);
    desc.SetAttr("use_mkldnn", false);
    desc.SetAttr("with_quant_attr", false);
    desc.Flush();
  };

  if (cache->count(var_node) == 0) {
    // insert cast op before node.
    std::string cast_input_name = var_node->Var()->Name();
    std::string cast_output_name =
        var_node->Var()->Name() + "_cast.tmp_" + std::to_string((*suffix)++);
    CHECK_NOTNULL(block_desc);
    framework::OpDesc cast_op_desc(block_desc);
    update_cast_desc(cast_op_desc,
                     cast_input_name,
                     cast_output_name,
                     static_cast<int>(from_type),
                     static_cast<int>(to_type));
    auto* cast_op_node = graph->CreateOpNode(&cast_op_desc);
    auto* cast_output_vardesc = block_desc->Var(cast_output_name);
    cast_output_vardesc->SetPersistable(false);
    cast_output_vardesc->SetDataType(to_type);
    cast_output_vardesc->SetShape(var_node->Var()->GetShape());
    auto* cast_output_node = graph->CreateVarNode(cast_output_vardesc);
    IR_NODE_LINK_TO(cast_op_node, cast_output_node);
    (*cache)[var_node] = cast_output_node;
  }
  op_node->Op()->Rename(var_node->Name(), cache->at(var_node)->Name());
  IR_NODE_LINK_TO(var_node, cache->at(var_node)->inputs[0]);
  IR_NODE_LINK_TO(cache->at(var_node), op_node);
}

inline bool IsFloatType(VarType::Type type) {
  return (type == VarType::FP64) || (type == VarType::FP32);
}

inline bool IsMixedType(VarType::Type type) {
  return (type == VarType::FP16) || (type == VarType::BF16);
}

};  // namespace

void FloatToMixedPass::Init(framework::ir::Graph* graph) const {
  CHECK_NOTNULL(graph);
  CHECK_EQ(graph->IsMainGraph(), true);

  auto graph_size = graph->SubGraphsSize();
  subgraphes_.resize(graph_size);
  all_nodes_.resize(graph_size);

  for (size_t i = 0; i < graph_size; i++) {
    subgraphes_[i] = graph->GetSubGraph(i);
    all_nodes_[i] = framework::ir::TopologySortOperations(*subgraphes_[i]);
    for (auto* var_node : all_nodes_[i]) {
      if (!var_node->IsVar()) continue;
      auto var_name = var_node->Var()->Name();
      if (real_vars_.count(var_name) == 0) {
        real_vars_[var_name] = var_node;
      }
    }
  }
}

void FloatToMixedPass::ApplyImpl(framework::ir::Graph* graph) const {
  Init(graph);

  SetOpUniqueType();

  GetVarInputOps();

  GetOpPrecision();

  SetVarAndUpdateOpPrecision();

  ProcessOpWithDtypeAttr();

  ProcessPersistableVar();

  InsertCastOp();

  RestoreOpOriginType();
}

bool FloatToMixedPass::OpSupportPrecision(const std::string& op_type,
                                          phi::DataType precision,
                                          phi::Backend backend) const {
  bool support = false;
  if (blacklist_.count(op_type) == 0) {
    if (backend == phi::Backend::GPU) {
      support = GpuKernelSupportPrecision(op_type, precision);
    }
  }
  return support;
}

void FloatToMixedPass::SetOpUniqueType() const {
  int suffix = 0;
  for (const auto& nodes : all_nodes_) {
    for (auto* op_node : nodes) {
      if (!op_node->IsOp()) continue;
      std::string unique_type =
          op_node->Op()->Type() + "_" + std::to_string(suffix++);
      op_original_type_[unique_type] = op_node->Op()->Type();
      op_node->Op()->SetType(unique_type);
      op_node->Op()->Flush();
    }
  }
}

void FloatToMixedPass::RestoreOpOriginType() const {
  for (const auto& nodes : all_nodes_) {
    for (auto* op_node : nodes) {
      if (!op_node->IsOp()) continue;
      if (op_original_type_.count(op_node->Op()->Type())) {
        op_node->Op()->SetType(op_original_type_[op_node->Op()->Type()]);
        op_node->Op()->Flush();
      }
    }
  }
}

void FloatToMixedPass::GetVarInputOps() const {
  for (const auto& nodes : all_nodes_) {
    for (auto* op_node : nodes) {
      if (!op_node->IsOp()) continue;
      for (auto* var_node : op_node->outputs) {
        CHECK_EQ(var_node->IsVar(), true);
        auto* real_var_node = real_vars_[var_node->Var()->Name()];
        if (real_var_node->Var()->Persistable()) continue;
        var_input_ops_[real_var_node->Var()->Name()].push_back(op_node);
      }
    }
  }
}

void FloatToMixedPass::ProcessOpWithDtypeAttr() const {
  for (const auto& nodes : all_nodes_) {
    for (auto* op_node : nodes) {
      if (!op_node->IsOp()) continue;

      auto dtype = op_node->Op()->GetAttrIfExists<int>("dtype");
      if (IsFloatType(static_cast<VarType::Type>(dtype)) &&
          op_run_mixed_[op_node->Op()->Type()]) {
        op_node->Op()->SetAttr(
            "dtype",
            static_cast<int>(framework::TransToProtoVarType(mixed_precision_)));
      }

      auto out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
      if (IsFloatType(static_cast<VarType::Type>(out_dtype)) &&
          op_run_mixed_[op_node->Op()->Type()]) {
        op_node->Op()->SetAttr(
            "out_dtype",
            static_cast<int>(framework::TransToProtoVarType(mixed_precision_)));
      }
    }
  }
}

void FloatToMixedPass::GetOpPrecision() const {
  for (const auto& nodes : all_nodes_) {
    for (auto* op_node : nodes) {
      if (!op_node->IsOp()) continue;

      bool support_mixed = OpSupportPrecision(
          op_original_type_[op_node->Op()->Type()], mixed_precision_);

      if (op_node->Op()->HasAttr("dtype")) {
        auto dtype = op_node->Op()->GetAttrIfExists<int>("dtype");
        support_mixed =
            support_mixed && IsFloatType(static_cast<VarType::Type>(dtype));
      } else if (op_node->Op()->HasAttr("out_dtype")) {
        auto out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
        support_mixed =
            support_mixed && IsFloatType(static_cast<VarType::Type>(out_dtype));
      } else {
        for (auto* var_node : op_node->inputs) {
          CHECK_EQ(var_node->IsVar(), true);
          auto* real_var_node = real_vars_[var_node->Var()->Name()];
          if (real_var_node->Var()->Persistable()) continue;
          support_mixed =
              support_mixed && IsFloatType(real_var_node->Var()->GetDataType());
        }
      }

      op_run_mixed_[op_node->Op()->Type()] = support_mixed;
    }
  }
}

void FloatToMixedPass::SetVarAndUpdateOpPrecision() const {
  bool precision_updated = false;
  do {
    precision_updated = false;
    for (const auto& nodes : all_nodes_) {
      for (auto* var_node : nodes) {
        if (!var_node->IsVar()) continue;
        auto* real_var_node = real_vars_[var_node->Var()->Name()];

        const auto& input_op_nodes =
            var_input_ops_[real_var_node->Var()->Name()];
        size_t mixed_num = 0;
        for (auto* op_node : input_op_nodes) {
          CHECK_EQ(op_node->IsOp(), true);
          if (op_run_mixed_[op_node->Op()->Type()]) {
            mixed_num++;
          }
        }
        if (mixed_num >= input_op_nodes.size() &&
            IsFloatType(real_var_node->Var()->GetDataType())) {
          real_var_node->Var()->SetDataType(
              framework::TransToProtoVarType(mixed_precision_));
          if (real_var_node->Var()->Persistable()) {
            weights_should_be_mixed_.insert(real_var_node->Var()->Name());
          }
        } else if (mixed_num > 0) {
          for (auto* op_node : input_op_nodes) {
            CHECK_EQ(op_node->IsOp(), true);
            op_run_mixed_[op_node->Op()->Type()] = false;
            precision_updated = true;
          }
        } else if (mixed_num == 0 &&
                   IsMixedType(real_var_node->Var()->GetDataType())) {
          real_var_node->Var()->SetDataType(VarType::FP32);
          if (weights_should_be_mixed_.count(real_var_node->Var()->Name())) {
            weights_should_be_mixed_.erase(real_var_node->Var()->Name());
          }
        }
      }
    }
  } while (precision_updated);
}

void FloatToMixedPass::ProcessPersistableVar() const {
  auto* scope = param_scope();
  auto var_names = scope->LocalVarNames();
  for (const auto& var_name : var_names) {
    if (weights_should_be_mixed_.count(var_name)) {
#define CONVERT_TENSOR_DTYPE(DTYPE, dtype)                                   \
  mixed_tensor.set_type(DTYPE);                                              \
  auto* mixed_data = mixed_tensor.mutable_data<dtype>(platform::CPUPlace()); \
  for (int64_t i = 0; i < origin_tensor->numel(); i++) {                     \
    mixed_data[i] = static_cast<dtype>(origin_data[i]);                      \
  }                                                                          \
  origin_tensor->clear();                                                    \
  paddle::framework::TensorCopySync(                                         \
      mixed_tensor, platform::CPUPlace(), origin_tensor)

      auto* var = scope->FindLocalVar(var_name);
      if (var->IsType<phi::DenseTensor>()) {
        auto* origin_tensor = var->GetMutable<phi::DenseTensor>();
        phi::DenseTensor mixed_tensor;
        mixed_tensor.Resize(origin_tensor->dims());
        auto* origin_data =
            origin_tensor->mutable_data<float>(platform::CPUPlace());
        if (mixed_precision_ == phi::DataType::FLOAT16) {
          CONVERT_TENSOR_DTYPE(paddle::experimental::DataType::FLOAT16,
                               phi::dtype::float16);
        } else if (mixed_precision_ == phi::DataType::BFLOAT16) {
          CONVERT_TENSOR_DTYPE(paddle::experimental::DataType::BFLOAT16,
                               phi::dtype::bfloat16);
        }
      }
    }
#undef CONVERT_TENSOR_DTYPE
  }
}

void FloatToMixedPass::InsertCastOp() const {
  // todo: 处理op具有相同输入输出var的情况

  int suffix = 0;
  std::unordered_map<framework::ir::Node*, framework::ir::Node*> cache;

  for (size_t i = 0; i < all_nodes_.size(); i++) {
    for (auto* op_node : all_nodes_[i]) {
      if (!op_node->IsOp()) continue;
      for (auto* var_node : op_node->inputs) {
        auto* real_var_node = real_vars_[var_node->Var()->Name()];
        if (IsFloatType(real_var_node->Var()->GetDataType()) &&
            op_run_mixed_[op_node->Op()->Type()]) {
          DoInsertCastOp(subgraphes_[i],
                         var_node,
                         op_node,
                         real_var_node->Var()->GetDataType(),
                         framework::TransToProtoVarType(mixed_precision_),
                         op_node->Op()->Block(),
                         &suffix,
                         &cache);
        } else if (IsMixedType(real_var_node->Var()->GetDataType()) &&
                   !op_run_mixed_[op_node->Op()->Type()]) {
          DoInsertCastOp(subgraphes_[i],
                         var_node,
                         op_node,
                         real_var_node->Var()->GetDataType(),
                         VarType::FP32,
                         op_node->Op()->Block(),
                         &suffix,
                         &cache);
        }
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(float_to_mixed_pass, paddle::framework::ir::FloatToMixedPass);
