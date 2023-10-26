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

#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_lowering_pass.h"

#include <unordered_map>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/op_with_group_merge_pass.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

namespace cinn {
namespace dialect {
namespace ir {

std::vector<pir::Value> GetBlockOutsideInput(
    const std::vector<pir::Operation*> op_list) {
  std::vector<pir::Value> vec_res;
  std::unordered_set<::pir::Value> block_inner_output;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_results(); ++i) {
      block_inner_output.insert(op_list[k]->result(i));
    }
  }

  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_operands(); ++i) {
      if (!block_inner_output.count(op_list[k]->operand_source(i))) {
        vec_res.push_back(op_list[k]->operand_source(i));
      }
    }
  }

  return vec_res;
}

std::vector<pir::Value> GetBlockOutsideOutput(
    const std::vector<pir::Operation*> op_list) {
  std::vector<pir::Value> vec_res;
  std::unordered_set<::pir::Value> block_inner_output;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_operands(); ++i) {
      block_inner_output.insert(op_list[k]->operand_source(i));
    }
  }

  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_results(); ++i) {
      if (!block_inner_output.count(op_list[k]->result(i))) {
        vec_res.push_back(op_list[k]->result(i));
      }
    }
  }

  return vec_res;
}

std::vector<pir::Operation*> GetOpListNotIncludeYield(
    const std::vector<pir::Operation*>& op_list) {
  std::vector<pir::Operation*> vec_res;
  for (size_t i = 0; i < op_list.size(); ++i) {
    if (!op_list[i]->isa<pir::YieldOp>()) {
      vec_res.push_back(op_list[i]);
    }
  }

  return vec_res;
}

std::unique_ptr<pir::Program> CINNGroupLoweringPass(::pir::Program* program) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

  std::string jit_op_name = cinn::dialect::JitKernelOp::name();
  ::pir::OpInfo op_info = ctx->GetRegisteredOpInfo(jit_op_name);

  auto ir_program = std::make_unique<::pir::Program>(ctx);
  std::unordered_map<pir::Value, pir::Value> value_map;
  std::vector<cinn::hlir::framework::PIRCompiler*> compiler_list;

  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, *program);

  for (auto it = program->block()->begin(); it != program->block()->end();
       ++it) {
    if ((*it)->isa<cinn::dialect::GroupOp>()) {
      // GetOpList and Call cinn CodeGen
      auto group_op = (*it)->dyn_cast<cinn::dialect::GroupOp>();

      // op fusion
      auto op_fusion = cinn::dialect::ir::OpFusionPassInternal(
          GetOpListNotIncludeYield(group_op.ops()));

      // fusion merge
      auto group_list =
          cinn::dialect::ir::GeneralFusionMergePassInternal(op_fusion);

      PADDLE_ENFORCE_EQ(group_list.size(),
                        1u,
                        phi::errors::Unimplemented(
                            "Only support one group after group fusion"));
      for (auto group : group_list) {
        auto ir_compiler =
            new cinn::hlir::framework::PIRCompiler(*program, target, scope);
        auto group1 =
            std::make_shared<cinn::hlir::framework::pir::Group>(group->nodes);
        auto fn_ptr_res = ir_compiler->BuildCUDAJITInfo({group1});
        compiler_list.push_back(ir_compiler);
        std::unordered_map<std::string, ::pir::Attribute> op_attrs{
            {cinn::dialect::JitKernelOp::kAttrName,
             cinn::dialect::CUDAJITInfoAttribute::get(ctx, fn_ptr_res[0])},
        };

        // Generate jit kernel op input and output
        auto vec_ins = GetBlockOutsideInput(group->nodes);

        std::vector<pir::Value> vec_new_ins;
        for (size_t i = 0; i < vec_ins.size(); ++i) {
          vec_new_ins.push_back(value_map.at(vec_ins[i]));
        }

        auto vec_outs = GetBlockOutsideOutput(group->nodes);

        std::vector<pir::Type> vec_types;
        for (auto& out : vec_outs) {
          vec_types.push_back(out.type());
        }

        ::pir::Operation* cinn_op =
            ::pir::Operation::Create(vec_new_ins, op_attrs, vec_types, op_info);

        // for (size_t i = 0; i < vec_outs.size(); ++i) {
        //   value_map[vec_outs[i]] = cinn_op->result(i);
        // }

        // auto yield_op = group_op.ops().back()->dyn_cast<pir::YieldOp>();
        for (size_t i = 0; i < group_op.num_results(); ++i) {
          value_map[group_op.result(i)] = cinn_op->result(i);
        }

        ir_program->block()->push_back(cinn_op);
      }

    } else {
      std::vector<pir::Value> vec_ins;

      for (size_t i = 0; i < (*it)->num_operands(); ++i) {
        vec_ins.push_back(value_map.at((*it)->operand_source(i)));
      }

      std::vector<pir::Type> vec_types;
      for (size_t i = 0; i < (*it)->num_results(); ++i) {
        vec_types.push_back((*it)->result(i).type());
      }

      ::pir::OpInfo info1 = ctx->GetRegisteredOpInfo((*it)->name());
      ::pir::Operation* op = ::pir::Operation::Create(
          vec_ins, (*it)->attributes(), vec_types, info1);

      ir_program->block()->push_back(op);

      value_map[(*it)->result(0)] = op->result(0);
    }
  }
  return ir_program;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
