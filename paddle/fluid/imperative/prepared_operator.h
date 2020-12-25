// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace framework {
class Tensor;
class Variable;
}  // namespace framework
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace imperative {

const framework::Tensor* GetTensorFromVar(const framework::Variable& var);

template <typename VarType>
static void SetForwardDataTypeOfGradVar(const std::shared_ptr<VarType>& var);

template <>
void SetForwardDataTypeOfGradVar<VariableWrapper>(
    const std::shared_ptr<VariableWrapper>& var) {
  if (var->HasGradVar()) {
    auto grad_var = var->GetGradVar();
    VLOG(6) << "Set grad var (" << grad_var->Name() << ") dtype to ("
            << framework::DataTypeToString(var->DataType()) << ").";
    grad_var->SetForwardDataType(var->DataType());
  }
}

template <>
void SetForwardDataTypeOfGradVar<VarBase>(const std::shared_ptr<VarBase>& var) {
  if (var->HasGradVar()) {
    auto& shared_var = var->SharedVar();
    SetForwardDataTypeOfGradVar<VariableWrapper>(shared_var);
  }
}

#ifdef PADDLE_WITH_XPU
static void ReplaceXPUKernelIfNotExists(
    const framework::OperatorWithKernel& op,
    framework::OpKernelType* expected_kernel_key) {
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());
  PADDLE_ENFORCE_NE(
      kernels_iter, all_op_kernels.end(),
      platform::errors::NotFound(
          "There are no kernels which are registered in the %s operator.",
          op.Type()));

  auto& kernels = kernels_iter->second;
  auto kernel_iter = kernels.find(*expected_kernel_key);
  if (kernel_iter == kernels.end() &&
      is_xpu_place(expected_kernel_key->place_)) {
    expected_kernel_key->place_ = platform::CPUPlace();
  }
}
#endif

template <typename VarType>
framework::OpKernelType GetExpectedKernelKey(
    const NameVarMap<VarType>& ins, const NameVarMap<VarType>& outs,
    const framework::OperatorWithKernel& op, const platform::Place& place,
    const framework::AttributeMap& attrs) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);
  framework::RuntimeContext ctx({}, {});

#ifdef PADDLE_WITH_MKLDNN
  // MKLDNN variant of code reads attributes in some of GetKernelTypeForVar and
  // GetKernelType functions, so we need to copy the attributes there.
  // Const qualifier of Attrs had to be discarded to overwrite it.
  if (FLAGS_use_mkldnn) {
    auto& mutable_op_attrs = const_cast<framework::AttributeMap&>(op.Attrs());
    mutable_op_attrs = attrs;
  }
#endif

  auto expected_kernel_key =
      op.GetExpectedKernelType(DygraphExecutionContext<VarType>(
          op, framework::Scope(), *dev_ctx, ctx, ins, outs, attrs));
#ifdef PADDLE_WITH_XPU
  ReplaceXPUKernelIfNotExists(op, &expected_kernel_key);
#endif
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  return expected_kernel_key;
}

template <typename VarType>
NameVarMap<VarType> PrepareData(
    const framework::OperatorWithKernel& op, const NameVarMap<VarType>& ins,
    const framework::OpKernelType& expected_kernel_key) {
  NameVarMap<VarType> tmp_ins(ins);
  for (auto& name_pair : tmp_ins) {
    for (auto& var_base : name_pair.second) {
      const auto* tensor = GetTensorFromVar(var_base->Var());
      SetForwardDataTypeOfGradVar(var_base);
      if (tensor && tensor->IsInitialized()) {
        auto kernel_type_for_var = op.GetKernelTypeForVar(
            name_pair.first, *tensor, expected_kernel_key);
        if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
          continue;
        } else {
          VLOG(3) << "Transform Variable " << var_base->Name() << " from "
                  << kernel_type_for_var << " to " << expected_kernel_key;
          framework::Tensor out;
          auto tmp_var = std::make_shared<VarType>(var_base->Name());
          tmp_var->SetType(var_base->Type());
          TransformData(expected_kernel_key, kernel_type_for_var, *tensor,
                        &out);
          SetTensorToVariable(var_base->Var(), out, tmp_var->MutableVar());
          var_base = tmp_var;
        }
      }
    }
  }
  return tmp_ins;
}

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const framework::OpKernelType& kernel_type,
             const framework::OperatorWithKernel::OpKernelFunc& func,
             platform::DeviceContext* dev_ctx);

  static PreparedOp Prepare(const framework::OperatorWithKernel& op,
                            const framework::OpKernelType& expected_kernel_key);

  void Run(const NameVarMap<VarBase>& in, const NameVarMap<VarBase>& out,
           const framework::AttributeMap& attrs);

  void Run(const NameVarMap<VariableWrapper>& ins,
           const NameVarMap<VariableWrapper>& outs,
           const framework::AttributeMap& attrs);

 private:
  const framework::OperatorBase& op_;
  const framework::RuntimeContext& ctx_;
  framework::OpKernelType kernel_type_;
  framework::OperatorWithKernel::OpKernelFunc func_;
  platform::DeviceContext* dev_ctx_;
};

}  // namespace imperative
}  // namespace paddle
