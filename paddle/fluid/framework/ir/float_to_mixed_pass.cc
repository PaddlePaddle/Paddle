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

inline bool IsFloatType(VarType::Type type) {
  return (type == VarType::FP64) || (type == VarType::FP32);
}

inline bool IsHalfType(VarType::Type type) {
  return (type == VarType::FP16) || (type == VarType::BF16);
}

};  // namespace

void FloatToMixedPass::Init(framework::ir::Graph* graph) const {
  //
  CHECK_NOTNULL(graph);
  CHECK_EQ(graph->IsMainGraph(), true);

  all_nodes_ = framework::ir::TopologySortOperations(*graph);
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
  for (auto* op_node : all_nodes_) {
    if (!op_node->IsOp()) continue;
    std::string unique_type =
        op_node->Op()->Type() + "_" + std::to_string(suffix++);
    op_original_type_[unique_type] = op_node->Op()->Type();
    op_node->Op()->SetType(unique_type);
    op_node->Op()->Flush();
  }
}

void FloatToMixedPass::RestoreOpOriginType() const {
  for (auto* op_node : all_nodes_) {
    if (!op_node->IsOp()) continue;
    if (op_original_type_.count(op_node->Op()->Type())) {
      op_node->Op()->SetType(op_original_type_[op_node->Op()->Type()]);
      op_node->Op()->Flush();
    }
  }
}

void FloatToMixedPass::GetVarInputOps() const {
  for (auto* op_node : all_nodes_) {
    if (!op_node->IsOp()) continue;
    for (auto* var_node : op_node->outputs) {
      CHECK_EQ(var_node->IsVar(), true);
      if (var_node->Var()->Persistable()) continue;
      var_input_ops_[var_node->Var()->Name()].insert(op_node);
    }
  }
}

void FloatToMixedPass::ProcessOpWithDtypeAttr() const {
  for (auto* op_node : all_nodes_) {
    if (!op_node->IsOp()) continue;

    auto dtype = op_node->Op()->GetAttrIfExists<int>("dtype");
    if (IsFloatType(static_cast<VarType::Type>(dtype)) &&
        op_run_half_[op_node->Op()->Type()]) {
      op_node->Op()->SetAttr(
          "dtype",
          static_cast<int>(framework::TransToProtoVarType(half_precision_)));
    }

    auto out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
    if (IsFloatType(static_cast<VarType::Type>(out_dtype)) &&
        op_run_half_[op_node->Op()->Type()]) {
      op_node->Op()->SetAttr(
          "out_dtype",
          static_cast<int>(framework::TransToProtoVarType(half_precision_)));
    }
  }
}

void FloatToMixedPass::GetOpPrecision() const {
  for (auto* op_node : all_nodes_) {
    if (!op_node->IsOp()) continue;
    bool support_half = OpSupportPrecision(
        op_original_type_[op_node->Op()->Type()], half_precision_);

    if (op_node->Op()->HasAttr("dtype")) {
      auto dtype = op_node->Op()->GetAttrIfExists<int>("dtype");
      support_half =
          support_half && IsFloatType(static_cast<VarType::Type>(dtype));
    } else if (op_node->Op()->HasAttr("out_dtype")) {
      auto out_dtype = op_node->Op()->GetAttrIfExists<int>("out_dtype");
      support_half =
          support_half && IsFloatType(static_cast<VarType::Type>(out_dtype));
    } else {
      for (auto* var_node : op_node->inputs) {
        CHECK_EQ(var_node->IsVar(), true);
        if (var_node->Var()->Persistable()) continue;
        support_half =
            support_half && IsFloatType(var_node->Var()->GetDataType());
      }
    }

    op_run_half_[op_node->Op()->Type()] = support_half;
  }
}

void FloatToMixedPass::SetVarAndUpdateOpPrecision() const {
  bool precision_updated = false;
  do {
    precision_updated = false;
    for (auto* var_node : all_nodes_) {
      if (!var_node->IsVar() || var_node->Var()->Persistable()) continue;
      var_node->GraphId();
      const auto& input_op_nodes = var_input_ops_[var_node->Var()->Name()];
      size_t half_num = 0;
      for (auto* op_node : input_op_nodes) {
        CHECK_EQ(op_node->IsOp(), true);
        if (op_run_half_[op_node->Op()->Type()]) {
          half_num++;
        }
      }
      if (half_num >= input_op_nodes.size() &&
          IsFloatType(var_node->Var()->GetDataType())) {
        var_node->Var()->SetDataType(
            framework::TransToProtoVarType(half_precision_));
      } else if (half_num > 0) {
        for (auto* op_node : input_op_nodes) {
          CHECK_EQ(op_node->IsOp(), true);
          op_run_half_[op_node->Op()->Type()] = false;
          precision_updated = true;
        }
      } else if (half_num == 0 && IsHalfType(var_node->Var()->GetDataType())) {
        var_node->Var()->SetDataType(VarType::FP32);
      }
    }
  } while (precision_updated);
}

void FloatToMixedPass::ProcessPersistableVar() const {
  //
}

void FloatToMixedPass::InsertCastOp() const {
  //
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(float_to_mixed_pass, paddle::framework::ir::FloatToMixedPass);
