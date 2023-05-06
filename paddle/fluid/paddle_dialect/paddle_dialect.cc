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

#include "paddle/fluid/paddle_dialect/paddle_dialect.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/ir/dialect_interface.h"
#include "paddle/ir/parameter.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace dialect {
class ParameterConvertInterface
    : public ir::DialectInterface::Base<ParameterConvertInterface> {
 public:
  explicit ParameterConvertInterface(ir::Dialect* dialect) : Base(dialect) {}

  paddle::framework::Variable* ParameterToVariable(ir::Parameter* parameter) {
    // if (parameter.type == ir::DenseTensorType) {
    //   new 一个 DenseTensor, 初始化一个 Variable并返回
    // } else {
    //   return nullptr;
    // }
    return nullptr;
  }

  ir::Parameter* VariableToParameter(paddle::framework::Variable* var) {
    // if (var->IsType<phi::DenseTensor>()) {
    //   new 一个 Parameter，返回这个指针
    // } else {
    //   return nullptr;
    // }
    return nullptr;
  }
};

PaddleDialect::PaddleDialect(ir::IrContext* context)
    : ir::Dialect(name(), context, ir::TypeId::get<PaddleDialect>()) {
  initialize();
}

void PaddleDialect::initialize() {
  // RegisterTypes<GET_BUILT_IN_TYPE_LIST>();
  // RegisterAttributes<GET_BUILT_IN_ATTRIBUTE_LIST>();
  // RegisterOps<GET_BUILT_IN_OP_LIST>();
  RegisterInterfaces<ParameterConvertInterface>();
}

}  // namespace dialect
}  // namespace paddle
