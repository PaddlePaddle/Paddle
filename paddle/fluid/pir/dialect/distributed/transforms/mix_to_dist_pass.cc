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
#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"

using paddle::dialect::DistDenseTensorType;

namespace paddle {
namespace dialect {

inline bool IsShardTensorOp(pir::Operation* op) {
  std::string op_name = op->name();
  return op_name.find("shard_tensor") != op_name.npos;
}

void ProcessBlock(pir::Block* block) {
  std::vector<pir::Operation*> deleted_ops;

  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    VLOG(0) << "main loop over op name " << op_item->name();

    if (paddle::dialect::IsShardTensorOp(op_item)) {
      pir::Value shard_operand_value = op_item->operand_source(0);
      pir::Value shard_result_value = op_item->result(0);
      pir::Operation* shard_operand_define_op =
          shard_operand_value.defining_op();
      std::string define_op_name = shard_operand_define_op->name();

      VLOG(0) << "here1";
      // TODO(2024-Q2) Support more paddle op
      if (define_op_name != "builtin.parameter" &&
          define_op_name != "pd_op.data") {
        PADDLE_THROW(platform::errors::Unimplemented(
            "op [%s] is not Supported by shard_tensor op in pir mode.",
            define_op_name));
      }
      VLOG(0) << "here2";
      // TODO(2024-Q2) Support shard_tensor is called after tensor has been
      // used.
      if (shard_operand_value.use_count() != 1) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "shard_tensor is supposed to be called right after tensor is "
            "created, the use_count of tensor to be sharded is [%d] which is "
            "not Supported in right now.",
            shard_operand_value.use_count()));
      }
      VLOG(0) << "here3";
      shard_operand_value.set_type(shard_result_value.type());
      VLOG(0) << "here4";
      shard_result_value.ReplaceAllUsesWith(shard_operand_value);
      VLOG(0) << "here5";
      // OperationDistAttribute op_dist_attr =
      //     op_item->attribute(kAttrOpDistAttrs)
      //         .dyn_cast<OperationDistAttribute>();
      // VLOG(0) << "here6";
      // VLOG(0) << "here6.1";
      // VLOG(0) << "here6.2";
      // OperationDistAttribute new_op_dist_attr =
      //     OperationDistAttribute::get(pir::IrContext::Instance(),
      //                                 op_dist_attr.process_mesh_attr(),
      //                                 op_dist_attr.operand_dist_attrs(),
      //                                 op_dist_attr.result_dist_attrs());
      VLOG(0) << "here7";
      shard_operand_define_op->set_attribute(
          kAttrOpDistAttrs, op_item->attribute(kAttrOpDistAttrs));
      VLOG(0) << "here8";
      deleted_ops.push_back(op_item);
    }

    // TODO(2024-Q2) Handle other shard annotation op in future.
  }
  VLOG(0) << "here8";
  for (auto* op : deleted_ops) {
    // TODO(2024-Q2) Support control flow / region
    op->Erase();
  }
  VLOG(0) << "here9";
}

/* Verification:
    1. all operators have OperatorDistAttr.
    2. all Values (Results) are DistDenseTensorType.
    3. no shard_tensor in block.
*/
void VerifyBlock(pir::Block* block) {
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    PADDLE_ENFORCE_EQ(paddle::dialect::IsShardTensorOp(op_item),
                      false,
                      phi::errors::PreconditionNotMet(
                          "Block still contain shard_tensor_op."));

    if (op_item && !op_item->HasAttribute(kAttrOpDistAttrs)) {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "The op [%s] does not hase OperatorDistAttr after Mix2Dist Pass.",
          op_item->name()));
    }

    for (size_t i = 0; i < op_item->num_results(); ++i) {
      PADDLE_ENFORCE_EQ(op_item->result(i).type().isa<DistDenseTensorType>(),
                        true,
                        phi::errors::PreconditionNotMet(
                            "[%d]'s input of [%s] is NOT DistDenseTensorType",
                            i,
                            op_item->name()));
    }

    VLOG(0) << "verifying op name " << op_item->name();
  }
}

std::shared_ptr<pir::Program> MixToDistPass(pir::Program* prog) {
  // if (FLAGS_print_ir) {
  std::cout << "IR before MixToDist Pass = " << *prog << std::endl;
  // }

  pir::IrMapping mapper;
  auto new_prog = prog->Clone(mapper);

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<DistDialect>();

  ProcessBlock(new_prog->block());
  VerifyBlock(new_prog->block());

  // if (FLAGS_print_ir) {
  std::cout << "IR after MixToDist Pass = " << *new_prog << std::endl;
  // }

  return new_prog;
}

}  // namespace dialect
}  // namespace paddle
