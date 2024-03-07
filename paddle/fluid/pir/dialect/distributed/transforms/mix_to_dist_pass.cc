// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/distributed/transforms/mix_to_dist_pass.h"

#include <iostream>
#include <unordered_set>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace dialect {

inline bool IsShardTensorOp(pir::Operation* op) {
  std::string op_name = op->name();
  return op_name.find("shard_tensor_op") != op_name.npos;
}

void ProcessBlock(pir::Block* block,
                  pir::Block* new_block,
                  pir::IrContext* ctx) {
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    VLOG(0) << "main loop over op name " << op_item->name();

    if (paddle::dialect::IsShardTensorOp(op_item)) {
      pir::Value shard_operand_value = op_item->operand_source(0);
      pir::Operation* shard_result_define_op =
          shard_operand_value.defining_op();
      std::string define_op_name = shard_result_define_op->name();

      // TODO(ljz) Support more paddle op
      if (define_op_name != "builtin.parameter" ||
          define_op_name != "pd_op.data") {
        PADDLE_THROW(platform::errors::Unimplemented(
            "op [%s] is not Supported by shard_tensor op in pir mode.",
            define_op_name));
      }

      // TODO(ljz) Support shard_tensor is called after tensor has been used.
      if (shard_operand_value.use_count() != 1) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "shard_tensor is supposed to be called right after tensor is "
            "created, the use_count of tensor to be sharded is [%d] which is "
            "not Supported in right now.",
            shard_operand_value.use_count()));
      }
    }

    // TODO(ljz) Handle other shard annotation op in future.
  }
}

/* Verification:
    1. all operators have OperatorDistAttr.
    2. all Values are DistDenseTensorType.
    3. no shard_tensor in block.
*/
void VerifyBlock(pir::Block* block) {
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    VLOG(0) << "verifying op name " << op_item->name();
  }
}

std::unique_ptr<pir::Program> MixToDistPass(pir::Program* prog) {
  // if (FLAGS_print_ir) {
  std::cout << "IR before MixToDist Pass = " << *prog << std::endl;
  // }

  auto program = std::make_unique<pir::Program>(pir::IrContext::Instance());

  auto block = prog->block();

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<DistDialect>();

  ProcessBlock(block, program->block(), ctx);

  VerifyBlock(program->block());

  // if (FLAGS_print_ir) {
  std::cout << "IR before MixToDist Pass = " << *program << std::endl;
  // }

  return program;
}

}  // namespace dialect
}  // namespace paddle
