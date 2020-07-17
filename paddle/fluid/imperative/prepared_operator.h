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
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace imperative {

const framework::Tensor* GetTensorFromVar(const framework::Variable& var);

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const framework::OperatorWithKernel::OpKernelFunc& func,
             platform::DeviceContext* dev_ctx);

  static PreparedOp Prepare(const NameVarMap<VarBase>& ins,
                            const NameVarMap<VarBase>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs);

  static PreparedOp Prepare(const NameVarMap<VariableWrapper>& ins,
                            const NameVarMap<VariableWrapper>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs);

  void Run(const NameVarMap<VarBase>& in, const NameVarMap<VarBase>& out,
           const framework::AttributeMap& attrs);

  void Run(const NameVarMap<VariableWrapper>& ins,
           const NameVarMap<VariableWrapper>& outs,
           const framework::AttributeMap& attrs);

 private:
  const framework::OperatorBase& op_;
  const framework::RuntimeContext& ctx_;
  framework::OperatorWithKernel::OpKernelFunc func_;
  platform::DeviceContext* dev_ctx_;
};

}  // namespace imperative
}  // namespace paddle
