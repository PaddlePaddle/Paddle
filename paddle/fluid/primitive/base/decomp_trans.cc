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

// #include <sstream>
// #include <string>

#include "paddle/fluid/primitive/base/decomp_trans.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/program.h"

namespace paddle {

using Program = pir::Program;

// static bool has_decomp_rule(const pir::Operation &op) {
//   pir::IrContext *ctx = pir::IrContext::Instance();
//   pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op.name());
//   auto decomp_interface_impl =
//       op_info.GetInterfaceImpl<paddle::dialect::DecompInterface>();
//   if (decomp_interface_impl == nullptr) return false;
//   return true;
// }

DecompProgram::DecompProgram(const pir::Program* program,
                             const std::vector<pir::OpResult>& src_vars)
    : program_(program), src_vars_(src_vars) {}

std::vector<pir::OpResult> DecompProgram::decomp_program() {
  std::ostringstream print_stream;
  program_->Print(print_stream);
  std::cout << "program in sink decomp.  " << print_stream.str() << std::endl;
  VLOG(0) << "sink decomp in ===========================";
  const pir::Block* block = program_->block();
  std::vector<pir::Operation*> ops_list;
  for (auto& op : *block) {
    ops_list.push_back(&op);
  }
  for (size_t i = 0; i < ops_list.size(); i++) {
    auto op = ops_list[i];
    bool flag = has_decomp_rule(*op);
    VLOG(0) << "op name ======= " << op->name();
    VLOG(0) << "decomp flag ======= " << flag;
  }
  return src_vars_;
}

}  // namespace paddle
