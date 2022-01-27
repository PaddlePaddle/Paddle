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

#include "paddle/fluid/eager/legacy/execution_context.h"
#include "paddle/fluid/eager/legacy/type_def.h"
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/type_defs.h"

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace pten {
class DenseTensor;
}  // namespace pten

namespace egr {
namespace legacy {

const paddle::framework::Tensor* GetTensorFromVar(
    const paddle::framework::Variable& var);

std::shared_ptr<NameTensorMap> PrepareData(
    const paddle::framework::OperatorWithKernel& op, const NameTensorMap& ins,
    const paddle::framework::OpKernelType& expected_kernel_key);

class PreparedOp {
 public:
  PreparedOp(const paddle::framework::OperatorBase& op,
             const paddle::framework::RuntimeContext& ctx,
             const paddle::framework::OpKernelType& kernel_type,
             const paddle::framework::OperatorWithKernel::OpKernelFunc& func,
             paddle::platform::DeviceContext* dev_ctx);

  static PreparedOp Prepare(
      const NameTensorMap& ins, const NameTensorMap& outs,
      const paddle::framework::OperatorWithKernel& op,
      const paddle::platform::Place& place,
      const paddle::framework::AttributeMap& attrs,
      const paddle::framework::AttributeMap& default_attrs);

  void Run(const NameTensorMap& in, const NameTensorMap& out,
           const paddle::framework::AttributeMap& attrs,
           const paddle::framework::AttributeMap& default_attrs);

  const paddle::framework::OpKernelType& kernel_type() const {
    return kernel_type_;
  }

 private:
  const paddle::framework::OperatorBase& op_;
  const paddle::framework::RuntimeContext& ctx_;
  paddle::framework::OpKernelType kernel_type_;
  paddle::framework::OperatorWithKernel::OpKernelFunc func_;
  paddle::platform::DeviceContext* dev_ctx_;
};

}  // namespace legacy
}  // namespace egr
