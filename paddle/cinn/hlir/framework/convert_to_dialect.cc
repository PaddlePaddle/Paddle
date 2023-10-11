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

#include "paddle/cinn/hlir/framework/convert_to_dialect.h"

#include <string>
#include <unordered_map>

#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/program.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/program.h"

namespace cinn {
namespace hlir {
namespace framework {

std::unique_ptr<::pir::Program> ConvertToRuntimeDialect(
    const hlir::framework::Program& program) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
  auto ir_program = std::make_unique<::pir::Program>(ctx);

  std::string jit_op_name = dialect::JitKernelOp::name();
  ::pir::OpInfo op_info = ctx->GetRegisteredOpInfo(jit_op_name);

  auto& instrs = program.GetRunInstructions();
  for (auto& instr : instrs) {
    std::unordered_map<std::string, ::pir::Attribute> op_attrs{
        {dialect::JitKernelOp::kAttrName,
         ::pir::PointerAttribute::get(ctx, instr.get())},
    };

    ::pir::Operation* cinn_op =
        ::pir::Operation::Create({}, op_attrs, {}, op_info);
    ir_program->block()->push_back(cinn_op);
  }
  return std::move(ir_program);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
