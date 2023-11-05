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
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/enforce.h"

namespace cinn {
namespace dialect {

const char* JitKernelOp::attributes_name[attributes_num] = {kAttrName};

void JitKernelOp::VerifySig() {
  VLOG(4) << "Verifying inputs, outputs and attributes for: JitKernelOp.";

  auto& attributes = this->attributes();

  IR_ENFORCE(
      attributes.count(kAttrName) > 0 &&
          attributes.at(kAttrName).isa<cinn::dialect::CUDAJITInfoAttribute>(),
      "Type of attribute: instruction is not right.");
}

const hlir::framework::pir::CUDAJITInfo& JitKernelOp::cuda_jit_info() {
  return attributes()
      .at(kAttrName)
      .dyn_cast<cinn::dialect::CUDAJITInfoAttribute>()
      .data();
}

}  // namespace dialect
}  // namespace cinn

IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::JitKernelOp)
