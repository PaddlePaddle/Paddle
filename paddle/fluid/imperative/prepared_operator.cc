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

platform::Place PreparedOp::GetExpectedPlace(const platform::Place& place,
                                             const NameVarBaseMap& ins) {
  bool found = false;
  for (auto& name_pair : ins) {
    for (auto& var_base : name_pair.second) {
      const auto* tensor = GetTensorFromVar(var_base->Var());
      if (tensor && tensor->IsInitialized()) {
        auto tmp_place = tensor->place();
        PADDLE_ENFORCE_EQ(!found || tmp_place == place, true,
                          "Input variable should keep in the same place: %s, "
                          "but get place: %s of input %s instead",
                          place, tmp_place, name_pair.first);
      }
    }
  }
  return place;
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

PreparedOp PreparedOp::Prepare(const framework::RuntimeContext& ctx,
                               const framework::OperatorWithKernel& op,
                               const platform::Place& place) {
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = op.AllOpKernels();
  auto kernels_iter = all_op_kernels.find(op.Type());
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.",
        op.Type());
  }

  auto& kernels = kernels_iter->second;

  auto expected_kernel_key =
      op.GetExpectedKernelType(framework::ExecutionContext(
          op, framework::Scope(), *dev_ctx, ctx, nullptr));
  VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);
  // TODO(jiabin): Add operator.cc's line 1000 part back when we need that case
  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("op %s does not have kernel for %s", op.Type(),
                 KernelTypeToString(expected_kernel_key));
  }
  std::vector<framework::KernelConfig>* kernel_configs =
      op.GetKernelConfig(expected_kernel_key);
  return PreparedOp(op, ctx, kernel_iter->second, dev_ctx, kernel_configs);
}

void PreparedOp::Run() {
  // TODO(zjl): remove scope in dygraph
  framework::Scope scope;
  op_.RuntimeInferShape(scope, dev_ctx_->GetPlace(), ctx_);
  func_(framework::ExecutionContext(op_, scope, *dev_ctx_, ctx_,
                                    kernel_configs_));
}

}  // namespace imperative
}  // namespace paddle
