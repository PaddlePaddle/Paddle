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

#include "paddle/fluid/imperative/prepared_operator.h"
#include <sstream>

namespace paddle {
namespace imperative {

const framework::Tensor* GetTensorFromVar(const framework::Variable& var) {
  if (var.IsType<framework::LoDTensor>()) {
    return &(var.Get<framework::LoDTensor>());
  } else if (var.IsType<framework::SelectedRows>()) {
    return &(var.Get<framework::SelectedRows>().value());
  } else {
    return nullptr;
  }
}

void PreparedOp::PrepareData(
    const platform::Place& place, const NameVarBaseMap& ins,
    const framework::OperatorWithKernel& op,
    const framework::OpKernelType& expected_kernel_key) {
  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      const auto* tensor = GetTensorFromVar(var_base->Var());
      if (tensor && tensor->IsInitialized()) {
        auto tmp_place = tensor->place();

        // TODO(jiabin): Support transform data layout when we Verify it on more
        // tests
        if (!(tmp_place == place)) {
          auto kernel_type_for_var = op.GetKernelTypeForVar(
              name_pair.first, *tensor, expected_kernel_key);
          if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
            continue;
          } else {
            VLOG(3) << "Transform Variable " << var_base->Name() << " from "
                    << kernel_type_for_var << " to " << expected_kernel_key;
            framework::Tensor out;
            TransformData(expected_kernel_key, kernel_type_for_var, *tensor,
                          &out);
            SetTensorToVariable(var_base->Var(), out, var_base->MutableVar());
          }
        }
      }
    }
  }
}

PreparedOp::PreparedOp(const framework::OperatorBase& op,
                       const framework::RuntimeContext& ctx,
                       framework::OperatorWithKernel::OpKernelFunc func,
                       platform::DeviceContext* dev_ctx,
                       std::vector<framework::KernelConfig>* kernel_configs)
    : op_(op),
      ctx_(ctx),
      func_(std::move(func)),
      dev_ctx_(dev_ctx),
      kernel_configs_(kernel_configs) {}

PreparedOp PreparedOp::Prepare(const NameVarBaseMap& ins,
                               const NameVarBaseMap& outs,
                               const framework::OperatorWithKernel& op,
                               platform::Place place,
                               const framework::AttributeMap* attrs) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.",
        op.Type());
  }

  auto& kernels = kernels_iter->second;

  framework::RuntimeContext ctx({}, {});
  auto expected_kernel_key = op.GetExpectedKernelType(DygraphExecutionContext(
      op, framework::Scope(), *dev_ctx, ctx, nullptr, ins, outs, attrs));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);
  // TODO(jiabin): Add operator.cc's line 1000 part back when we need that case
  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("op %s does not have kernel for %s", op.Type(),
                 KernelTypeToString(expected_kernel_key));
  }
  std::vector<framework::KernelConfig>* kernel_configs =
      op.GetKernelConfig(expected_kernel_key);

  if (!(expected_kernel_key.place_ == place)) {
    dev_ctx = pool.Get(expected_kernel_key.place_);
    place = dev_ctx->GetPlace();
  }

  PrepareData(place, ins, op, expected_kernel_key);
  return PreparedOp(op, ctx, kernel_iter->second, dev_ctx, kernel_configs);
}

void PreparedOp::Run(const NameVarBaseMap* in, const NameVarBaseMap* out,
                     const framework::AttributeMap* attrs) {
  // TODO(zjl): remove scope in dygraph
  framework::Scope scope;

  DygraphInferShapeContext infer_shape_ctx(in, out, attrs);

  framework::OperatorWithKernel* op_ker =
      (framework::OperatorWithKernel*)(&op_);

  op_ker->InferShape(&infer_shape_ctx);

  func_(DygraphExecutionContext(op_, scope, *dev_ctx_, ctx_, kernel_configs_,
                                *in, *out, attrs));
}

}  // namespace imperative
}  // namespace paddle
