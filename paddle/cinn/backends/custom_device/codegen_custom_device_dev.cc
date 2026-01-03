// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/custom_device/codegen_custom_device_dev.h"

namespace cinn {
namespace backends {
namespace custom_device {

const std::string CodeGenCustomDevice::source_header_ =  // NOLINT
    R"(#define CINN_WITH_CUSTOM_DEVICE
     #include "float16.h"
     using cinn::common::float16;
     #include "cinn_custom_device_runtime_source.h"
)";

const std::string &CodeGenCustomDevice::GetSourceHeader() {
  return source_header_;
}

CodeGenCustomDevice::CodeGenCustomDevice(Target target)
    : CodeGenGpuDev(target) {}

void CodeGenCustomDevice::PrintIncludes() { str_ += GetSourceHeader(); }

void CodeGenCustomDevice::Visit(const ir::Min *op) {
  str_ += "std::min(";
  ir::Expr a = op->a(), b = op->b();
  auto [unify_bit, both_dyn] =
      common::UnifiedOperandTypeBits(&this->DynamicShapeMap(), op);
  this->ProcessMinMaxOperand(&a, &b, unify_bit, both_dyn);
  IrPrinter::Visit(a);
  str_ += ", ";
  IrPrinter::Visit(b);
  str_ += ")";
}

void CodeGenCustomDevice::Visit(const ir::Max *op) {
  str_ += "std::max(";
  ir::Expr a = op->a(), b = op->b();
  auto [unify_bit, both_dyn] =
      common::UnifiedOperandTypeBits(&this->DynamicShapeMap(), op);
  this->ProcessMinMaxOperand(&a, &b, unify_bit, both_dyn);
  IrPrinter::Visit(a);
  str_ += ", ";
  IrPrinter::Visit(b);
  str_ += ")";
}

}  // namespace custom_device
}  // namespace backends
}  // namespace cinn
