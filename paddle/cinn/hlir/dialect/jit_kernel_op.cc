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

#include "paddle/cinn/hlir/dialect/jit_kernel_op.h"

#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/phi/core/enforce.h"

namespace cinn {
namespace dialect {

const char* JitKernelOp::attributes_name[attributes_num] = {kAttrName};

void JitKernelOp::Verify() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: JitKernelOp.";

  auto& attributes = this->attributes();

  PADDLE_ENFORCE(attributes.count(kAttrName) > 0 &&
                     attributes.at(kAttrName).isa<::ir::PointerAttribute>(),
                 phi::errors::PreconditionNotMet(
                     "Type of attribute: instruction is not right."));
}

hlir::framework::Instruction* JitKernelOp::instruction() {
  void* ptr =
      attributes().at(kAttrName).dyn_cast<ir::PointerAttribute>().data();
  return reinterpret_cast<hlir::framework::Instruction*>(ptr);
}

}  // namespace dialect
}  // namespace cinn

IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::JitKernelOp)
