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

#include "paddle/fluid/ir/dialect/kernel_op.h"
#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace dialect {

const char *PhiKernelOp::attributes_name[attributes_num] = {
    "op_name", "kernel_name", "kernel_key"};

void PhiKernelOp::Verify(const std::vector<ir::OpResult> &inputs,
                         const std::vector<ir::Type> &outputs,
                         const ir::AttributeMap &attributes) {
  VLOG(4) << "Verifying inputs, outputs and attributes for: PhiKernelOp.";
  // Verify if attributes contain attribute name in attributes_name:
  PADDLE_ENFORCE_EQ(attributes.count("op_name") > 0 &&
                        attributes.at("op_name").isa<ir::StrAttribute>(),
                    true,
                    phi::errors::PreconditionNotMet(
                        "Type of attribute: op_name is not right."));
  PADDLE_ENFORCE_EQ(attributes.count("kernel_name") > 0 &&
                        attributes.at("kernel_name").isa<ir::StrAttribute>(),
                    true,
                    phi::errors::PreconditionNotMet(
                        "Type of attribute: kernel_name is not right."));
  PADDLE_ENFORCE_EQ(attributes.count("kernel_key") > 0 &&
                        attributes.at("kernel_key").isa<KernelAttribute>(),
                    true,
                    phi::errors::PreconditionNotMet(
                        "Type of attribute: kernel_key is not right."));
}

const std::string PhiKernelOp::op_name() {
  return operation()
      ->attributes()
      .at("op_name")
      .dyn_cast<ir::StrAttribute>()
      .data();
}
const std::string PhiKernelOp::kernel_name() {
  return operation()
      ->attributes()
      .at("kernel_name")
      .dyn_cast<ir::StrAttribute>()
      .data();
}
phi::KernelKey PhiKernelOp::kernel_key() {
  return operation()
      ->attributes()
      .at("kernel_key")
      .dyn_cast<KernelAttribute>()
      .data();
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PhiKernelOp)
