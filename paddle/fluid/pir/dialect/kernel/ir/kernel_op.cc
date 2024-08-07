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

#include <glog/logging.h>

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace paddle::dialect {

const char* PhiKernelOp::attributes_name[attributes_num] = {  // NOLINT
    "op_name",
    "kernel_name",
    "kernel_key"};

void PhiKernelOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: PhiKernelOp.";

  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count("op_name") > 0 &&
                     attributes.at("op_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: op_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_name") > 0 &&
                     attributes.at("kernel_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_key") > 0 &&
                     attributes.at("kernel_key").isa<KernelAttribute>(),
                 common::errors::PreconditionNotMet(
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

void LegacyKernelOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: LegacyKernelOp.";

  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count("op_name") > 0 &&
                     attributes.at("op_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: op_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_name") > 0 &&
                     attributes.at("kernel_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_key") > 0 &&
                     attributes.at("kernel_key").isa<KernelAttribute>(),
                 common::errors::PreconditionNotMet(
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

const char* CustomKernelOp::attributes_name[attributes_num] = {  // NOLINT
    "op_name",
    "kernel_name",
    "kernel_key"};

void CustomKernelOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: CustomKernelOp.";
  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count("op_name") > 0 &&
                     attributes.at("op_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: op_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_name") > 0 &&
                     attributes.at("kernel_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_key") > 0 &&
                     attributes.at("kernel_key").isa<KernelAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_key is not right."));
}

std::string CustomKernelOp::op_name() {
  return attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
}

std::string CustomKernelOp::kernel_name() {
  return attributes()
      .at("kernel_name")
      .dyn_cast<pir::StrAttribute>()
      .AsString();
}

phi::KernelKey CustomKernelOp::kernel_key() {
  return attributes().at("kernel_key").dyn_cast<KernelAttribute>().data();
}

#ifdef PADDLE_WITH_DNNL
const char* OneDNNPhiKernelOp::attributes_name[attributes_num] = {  // NOLINT
    "op_name",
    "kernel_name",
    "kernel_key"};

void OneDNNPhiKernelOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: OneDNNPhiKernelOp.";

  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count("op_name") > 0 &&
                     attributes.at("op_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: op_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_name") > 0 &&
                     attributes.at("kernel_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_key") > 0 &&
                     attributes.at("kernel_key").isa<KernelAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_key is not right."));
}

std::string OneDNNPhiKernelOp::op_name() {
  return attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
}
std::string OneDNNPhiKernelOp::kernel_name() {
  return attributes()
      .at("kernel_name")
      .dyn_cast<pir::StrAttribute>()
      .AsString();
}

phi::KernelKey OneDNNPhiKernelOp::kernel_key() {
  return attributes().at("kernel_key").dyn_cast<KernelAttribute>().data();
}

const char* OneDNNMixedPhiKernelOp::attributes_name[attributes_num] =  // NOLINT
    {"op_name", "kernel_name", "kernel_key"};

void OneDNNMixedPhiKernelOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: "
             "OneDNNMixedPhiKernelOp.";

  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count("op_name") > 0 &&
                     attributes.at("op_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: op_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_name") > 0 &&
                     attributes.at("kernel_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_key") > 0 &&
                     attributes.at("kernel_key").isa<KernelAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_key is not right."));
}

std::string OneDNNMixedPhiKernelOp::op_name() {
  return attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
}
std::string OneDNNMixedPhiKernelOp::kernel_name() {
  return attributes()
      .at("kernel_name")
      .dyn_cast<pir::StrAttribute>()
      .AsString();
}
phi::KernelKey OneDNNMixedPhiKernelOp::kernel_key() {
  return attributes().at("kernel_key").dyn_cast<KernelAttribute>().data();
}

const char* OneDNNLegacyKernelOp::attributes_name[attributes_num] = {  // NOLINT
    "op_name",
    "kernel_name",
    "kernel_key"};

void OneDNNLegacyKernelOp::VerifySig() {
  VLOG(4)
      << "Verifying inputs, outputs and attributes for: OneDNNLegacyKernelOp.";

  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count("op_name") > 0 &&
                     attributes.at("op_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: op_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_name") > 0 &&
                     attributes.at("kernel_name").isa<pir::StrAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_name is not right."));

  PADDLE_ENFORCE(attributes.count("kernel_key") > 0 &&
                     attributes.at("kernel_key").isa<KernelAttribute>(),
                 common::errors::PreconditionNotMet(
                     "Type of attribute: kernel_key is not right."));
}

std::string OneDNNLegacyKernelOp::op_name() {
  return attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
}
std::string OneDNNLegacyKernelOp::kernel_name() {
  return attributes()
      .at("kernel_name")
      .dyn_cast<pir::StrAttribute>()
      .AsString();
}
phi::KernelKey OneDNNLegacyKernelOp::kernel_key() {
  return attributes().at("kernel_key").dyn_cast<KernelAttribute>().data();
}
#endif

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PhiKernelOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::LegacyKernelOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::CustomKernelOp)
#ifdef PADDLE_WITH_DNNL
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNPhiKernelOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNMixedPhiKernelOp)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::OneDNNLegacyKernelOp)
#endif
