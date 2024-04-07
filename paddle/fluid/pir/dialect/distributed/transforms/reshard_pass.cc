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

#include "paddle/fluid/pir/dialect/distributed/transforms/reshard_pass.h"
#include "paddle/fluid/pir/dialect/distributed/transforms/reshard_builder.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function_registry.h"

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
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"

REGISTER_FILE_SYMBOLS(reshard_pass);

using paddle::dialect::DistDenseTensorType;

COMMON_DECLARE_bool(print_ir);

namespace paddle {
namespace dialect {

inline bool IsReShardOp(pir::Operation* op) {
  std::string op_name = op->name();
  return op_name.find("reshard") != op_name.npos;
}

static void ProcessBlock(pir::Block* block) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Builder builder(ctx, block);
  std::vector<pir::Operation*> deleted_ops;

  for (auto iter = block->begin(); iter != block->end();) {
    pir::Operation* op_item = &(*iter);
    ++iter;
    VLOG(0) << "reshard main loop over op name " << op_item->name();

    if (paddle::dialect::IsReShardOp(op_item)) {
      // Trans pir::Value to DenseTensor/DistTensor
      pir::Value reshard_operand_value = op_item->operand_source(0);

      // TODO(ywt01) hack GPUPlace
      DistDenseTensorType dist_tensor_type =
          reshard_operand_value.type().dyn_cast<DistDenseTensorType>();

      // construct src TensorDistAttr from src TensorDistAttribute
      std::unique_ptr<phi::distributed::TensorDistAttr> src_dist_attr =
          std::make_unique<phi::distributed::TensorDistAttr>();
      TensorDistAttribute src_tensor_dist_attr =
          dist_tensor_type.tensor_dist_attr();
      src_dist_attr->set_process_mesh(
          src_tensor_dist_attr.process_mesh_attr().process_mesh());
      src_dist_attr->set_dims_mapping(src_tensor_dist_attr.dims_mapping());
      src_dist_attr->set_partial_status(src_tensor_dist_attr.partial_status());

      // auto dist_tensor =
      // std::make_unique<phi::distributed::DistTensor>(dense_tensor,
      // *src_dist_attr);
      auto dist_tensor = std::make_unique<phi::distributed::DistTensor>(
          TransToPhiDataType(dist_tensor_type.dtype()));
      dist_tensor->unsafe_set_dims(dist_tensor_type.global_ddim());
      dist_tensor->unsafe_set_dist_attr(*src_dist_attr);

      // construct dst TensorDistAttr from dst TensorDistAttribute
      TensorDistAttribute dst_tensor_dist_attr =
          op_item->attribute<OperationDistAttribute>(kAttrOpDistAttr)
              .result_dist_attr(0);
      std::unique_ptr<phi::distributed::TensorDistAttr> dst_dist_attr =
          std::make_unique<phi::distributed::TensorDistAttr>();
      dst_dist_attr->set_process_mesh(
          dst_tensor_dist_attr.process_mesh_attr().process_mesh());
      dst_dist_attr->set_dims_mapping(dst_tensor_dist_attr.dims_mapping());
      dst_dist_attr->set_partial_status(dst_tensor_dist_attr.partial_status());

      VLOG(0) << "src_dist_attr " << src_dist_attr->to_string();
      VLOG(0) << "dst_dist_attr " << dst_dist_attr->to_string();

      VLOG(0) << "find reshard op " << op_item->name();

      if (dist_tensor->dist_attr() != *dst_dist_attr) {
        VLOG(0) << "find diff dist_attr ";
        auto* func = phi::distributed::ChooseProperReshardFunction(
            *dist_tensor, *dst_dist_attr);
        func->Eval(nullptr, *dist_tensor, *dst_dist_attr);

        auto reshard_func_desc = func->GetReshardFuncDescs();
        auto func_size = reshard_func_desc.size();
        VLOG(0) << "find diff func_size " << func_size;

        pir::Operation* new_op;
        for (size_t i = 0; i < func_size; ++i) {
          if (i == 0) {
            builder.set_insertion_point(op_item);
            new_op = Build(reshard_func_desc[i].get(),
                           builder,
                           std::vector<pir::Value>({reshard_operand_value}));
            reshard_operand_value.ReplaceUsesWithIf(
                new_op->result(0),
                [new_op](pir::OpOperand op) { return op.owner() != new_op; });
            op_item->Erase();
          } else {
            reshard_operand_value = new_op->result(0);
            pir::Operation* new_op_tmp =
                Build(reshard_func_desc[i].get(),
                      builder,
                      std::vector<pir::Value>({reshard_operand_value}));
            reshard_operand_value.ReplaceUsesWithIf(
                new_op_tmp->result(0), [new_op_tmp](pir::OpOperand op) {
                  return op.owner() != new_op_tmp;
                });
            new_op = new_op_tmp;
          }
        }
      } else {
        VLOG(0) << "not find diff dist_attr ";
      }
    }
  }
}

/* Verification:
    1. all operators have OperatorDistAttr.
    2. all Values (Results) are DistDenseTensorType.
    3. no shard_tensor in block.
*/
static void VerifyBlock(pir::Block* block) {
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    PADDLE_ENFORCE_EQ(
        paddle::dialect::IsReShardOp(op_item),
        false,
        phi::errors::PreconditionNotMet("Block still contain reshard_op."));

    if (op_item && !op_item->HasAttribute(kAttrOpDistAttr)) {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "The op [%s] does not hase OperatorDistAttr after Reshard Pass.",
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
  }
}

std::shared_ptr<pir::Program> ReshardPass(pir::Program* prog) {
  VLOG(0) << "ReshardPass begin";
  if (FLAGS_print_ir) {
    std::cout << "IR before Reshard Pass = " << *prog << std::endl;
  }

  pir::IrMapping mapper;
  auto new_prog = prog->Clone(mapper);

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<DistDialect>();

  ProcessBlock(new_prog->block());
  // VerifyBlock(new_prog->block());

  if (FLAGS_print_ir) {
    std::cout << "IR after Reshard Pass = " << *new_prog << std::endl;
  }

  return new_prog;
}

}  // namespace dialect
}  // namespace paddle
