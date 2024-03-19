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

#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"

#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace cinn {
namespace dialect {

const char* JitKernelOp::attributes_name[attributes_num] = {kAttrName,
                                                            kKernelTensorNumber,
                                                            "input_dim_exprs",
                                                            "symbol_bindings",
                                                            "output_dim_exprs"};

void JitKernelOp::Build(::pir::Builder& builder,
                        pir::OperationArgument& argument,
                        const std::vector<::pir::Value>& x,
                        const ::pir::AttributeMap& attributes,
                        const std::vector<::pir::Type>& out_types) {
  VLOG(4) << "Start build JitKernelOp";

  VLOG(4) << "Builder construction inputs";
  argument.AddInputs(x);

  VLOG(4) << "Builder construction attributes";
  argument.AddAttributes(attributes);

  VLOG(4) << "Builder construction outputs";
  argument.AddOutputs(out_types.begin(), out_types.end());
}

void JitKernelOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: JitKernelOp.";

  auto& attributes = this->attributes();
  PADDLE_ENFORCE_EQ(attributes.count(kAttrName) > 0 &&
                        attributes.at(kAttrName)
                            .isa<cinn::dialect::CINNKernelInfoAttribute>(),
                    true,
                    "Type of attribute: instruction is not right.");

  PADDLE_ENFORCE_EQ(
      attributes.count(kKernelTensorNumber),
      true,
      "Type of attribute:  Kernel tensor number should in attribute map");
  auto kernel_tensor_number =
      attribute(kKernelTensorNumber).dyn_cast<pir::Int64Attribute>().data();

  PADDLE_ENFORCE_EQ(
      kernel_tensor_number > 0 && kernel_tensor_number <= this->num_operands(),
      true,
      "Kernel tensor number [%d] should > 0 and < [%d] input number",
      kernel_tensor_number,
      this->num_operands());
}

const hlir::framework::pir::CINNKernelInfo& JitKernelOp::cinn_kernel_info() {
  return attributes()
      .at(kAttrName)
      .dyn_cast<cinn::dialect::CINNKernelInfoAttribute>()
      .data();
}

int64_t JitKernelOp::kernel_tensor_number() {
  return attribute(kKernelTensorNumber).dyn_cast<pir::Int64Attribute>().data();
}

}  // namespace dialect
}  // namespace cinn

IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::JitKernelOp)
