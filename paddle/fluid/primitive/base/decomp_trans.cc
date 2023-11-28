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

bool has_decomp_rule(const pir::Operation& op) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op.name());
  auto decomp_interface_impl =
      op_info.GetInterfaceImpl<paddle::dialect::DecompInterface>();
  if (decomp_interface_impl == nullptr) return false;
  return true;
}

std::vector<std::vector<pir::OpResult>> call_decomp_rule(pir::Operation* op) {
  paddle::dialect::DecompInterface decomp_interface =
      op->dyn_cast<paddle::dialect::DecompInterface>();
  PADDLE_ENFORCE(
      decomp_interface,
      phi::errors::InvalidArgument(
          "The decomp function is not registered in %s op ", op->name()));
  std::vector<std::vector<pir::OpResult>> decomp_res =
      decomp_interface.Decomp(op);
  return decomp_res;
}

DecompProgram::DecompProgram(const pir::Program* program,
                             const std::vector<pir::OpResult>& src_vars)
    : program_(program), src_vars_(src_vars) {}

std::vector<pir::OpResult> DecompProgram::decomp_program() {
  // std::ostringstream print_stream;
  // program_->Print(print_stream);
  // VLOG(0) << "program in sink decomp ------" << print_stream.str();
  std::vector<pir::OpResult> tar_vars;
  VLOG(0) << "sink decomp in ===========================";
  pir::Block* block = const_cast<pir::Block*>(program_->block());
  std::vector<pir::Operation*> ops_list;
  for (auto& op : *block) {
    ops_list.push_back(&op);
  }
  for (size_t i = 0; i < ops_list.size(); i++) {
    auto op = ops_list[i];
    bool flag = has_decomp_rule(*op);
    if (flag) {
      std::vector<std::vector<pir::OpResult>> decomp_res = call_decomp_rule(op);
      VLOG(4) << "decomp out size ======= " << decomp_res.size();
      op->ReplaceAllUsesWith(decomp_res[0]);
      auto op_iter = std::find(block->begin(), block->end(), *op);
      block->erase(op_iter);
      tar_vars = decomp_res[0];
    }
    VLOG(4) << "op name ======= " << op->name();
    // std::ostringstream print_stream2;
    // program_->Print(print_stream2);
    // VLOG(4) << "program out sink decomp ------" << print_stream2.str();
    VLOG(4) << "decomp flag ======= " << flag;
  }
  return tar_vars;
}

}  // namespace paddle
