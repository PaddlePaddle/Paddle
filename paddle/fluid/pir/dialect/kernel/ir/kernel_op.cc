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

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/builtin_attribute.h"

namespace paddle {
namespace dialect {

const char* PhiKernelOp::attributes_name[attributes_num] = {  // NOLINT
    "op_name",
    "kernel_name",
    "kernel_key"};

void PhiKernelOp::Verify() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: PhiKernelOp.";

  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count("op_name") > 0 &&
                     attributes.at("op_name").isa<pir::StrAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "Type of attribute: op_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_name") > 0 &&
                     attributes.at("kernel_name").isa<pir::StrAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "Type of attribute: kernel_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_key") > 0 &&
                     attributes.at("kernel_key").isa<KernelAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "Type of attribute: kernel_key is not right."));
}

std::string PhiKernelOp::op_name() {
  return attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
}
std::string PhiKernelOp::kernel_name() {
  return attributes()
      .at("kernel_name")
      .dyn_cast<pir::StrAttribute>()
      .AsString();
}
phi::KernelKey PhiKernelOp::kernel_key() {
  return attributes().at("kernel_key").dyn_cast<KernelAttribute>().data();
}

const char* LegacyKernelOp::attributes_name[attributes_num] = {  // NOLINT
    "op_name",
    "kernel_name",
    "kernel_key"};

void LegacyKernelOp::Verify() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: LegacyKernelOp.";

  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count("op_name") > 0 &&
                     attributes.at("op_name").isa<pir::StrAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "Type of attribute: op_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_name") > 0 &&
                     attributes.at("kernel_name").isa<pir::StrAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "Type of attribute: kernel_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_key") > 0 &&
                     attributes.at("kernel_key").isa<KernelAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "Type of attribute: kernel_key is not right."));
}

std::string LegacyKernelOp::op_name() {
  return attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
}
std::string LegacyKernelOp::kernel_name() {
  return attributes()
      .at("kernel_name")
      .dyn_cast<pir::StrAttribute>()
      .AsString();
}
phi::KernelKey LegacyKernelOp::kernel_key() {
  return attributes().at("kernel_key").dyn_cast<KernelAttribute>().data();
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PhiKernelOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::LegacyKernelOp)
