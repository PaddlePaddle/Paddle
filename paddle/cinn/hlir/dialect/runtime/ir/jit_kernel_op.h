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

#pragma once

#include "paddle/cinn/hlir/framework/new_ir/utils.h"
#include "paddle/pir/core/op_base.h"

namespace cinn {

namespace dialect {

/*
 * TODO(Aurelius84): THIS IS NOT FINAL STATE!
 *   JitKernel is unified runtime operation to represent
 *   jit compiled function ptr from backend, such as
 *   nvrct.

 *   Ideally, JitKernel should only contains ArrayAttribute
 *   with each element is PointerAttribute, which is jit
 *   function ptr indeed.

 *   Currently, we regard hlir::framework::Instruction
 *   temporarily, and will spilt executor information like
 *   scope, inputs, outputs into InterpretorCore module.
*/
class JitKernelOp : public ::pir::Op<JitKernelOp> {
 public:
  using Op::Op;
  static const char* name() { return "cinn_runtime.jit_kernel"; }
  // TODO(Aurelius84): Think deeply what should contains
  static constexpr uint32_t attributes_num = 1;
  static constexpr char* kAttrName = "jit_info";
  static const char* attributes_name[attributes_num];

  const hlir::framework::newir::CUDAJITInfo& cuda_jit_info();

  void VerifySig();
};

}  // namespace dialect
}  // namespace cinn

IR_DECLARE_EXPLICIT_TYPE_ID(cinn::dialect::JitKernelOp)
