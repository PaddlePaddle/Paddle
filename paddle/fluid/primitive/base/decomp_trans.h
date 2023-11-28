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

#include <memory>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/pir/dialect/operator/interface/decomp.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/program.h"

namespace paddle {

class DecompProgram {
 public:
  DecompProgram(const pir::Program* program,
                const std::vector<pir::OpResult>& src_vars);

  std::vector<pir::OpResult> decomp_program();

 private:
  const pir::Program* program_;
  std::vector<pir::OpResult> src_vars_;
  std::vector<pir::OpResult> tar_vars_;
};

static bool has_decomp_rule(const pir::Operation& op) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op.name());
  auto decomp_interface_impl =
      op_info.GetInterfaceImpl<paddle::dialect::DecompInterface>();
  if (decomp_interface_impl == nullptr) return false;
  return true;
}

static std::vector<std::vector<pir::OpResult>> call_decomp_rule(
    pir::Operation* op) {
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

// pir::Program decomp_program(
//     const pir::Program& program);

}  // namespace paddle
