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
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"

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

  std::unordered_set<::pir::Value> insert_value;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_operands(); ++i) {
      if (!block_inner_output.count(op_list[k]->operand_source(i)) &&
          !insert_value.count(op_list[k]->operand_source(i))) {
        vec_res.push_back(op_list[k]->operand_source(i));
        insert_value.insert(op_list[k]->operand_source(i));
      }
    }
  }
  return vec_res;
}

std::vector<pir::Value> GetBlockOutsideOutput(
    const std::vector<pir::Operation*> op_list,
    const std::vector<pir::Operation*> group_all_list) {
  assert(group_all_list.size() >= 2);
  assert(group_all_list.back()->isa<pir::YieldOp>());

  auto yeild_op = group_all_list.back()->dyn_cast<pir::YieldOp>();

  std::unordered_set<pir::Value> yeild_inputs;
  for (size_t i = 0; i < yeild_op.num_operands(); ++i) {
    yeild_inputs.insert(yeild_op.operand_source(i));
  }

  std::unordered_set<pir::Operation*> innner_op_set(op_list.begin(),
                                                    op_list.end());
  std::unordered_set<pir::Operation*> outside_group_set;

  for (size_t i = 0; i < group_all_list.size(); ++i) {
    if (!innner_op_set.count(group_all_list[i])) {
      outside_group_set.insert(group_all_list[i]);
    }
  }

  std::vector<pir::Value> vec_res;

  for (auto* op : op_list) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      if (yeild_inputs.count(op->result(i))) {
        vec_res.push_back(op->result(i));
      } else {
        for (auto it = op->result(i).use_begin(); it != op->result(i).use_end();
             ++it) {
          if (outside_group_set.count(it->owner())) {
            vec_res.push_back(op->result(i));
            break;
          }
        }
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

std::shared_ptr<pir::ShapeConstraintIRAnalysis> CreateShapeAnalysis(
    const pir::Program* program) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();

  auto shape_analysis =
      std::make_shared<pir::MockShapeConstraintIRAnalysis>(ctx);
  pir::SymbolicDimMgr& sym_dim_mgr = shape_analysis->symbolicDimMgr();

  std::vector<pir::Value> candidate_values;

  for (auto it = program->block()->begin(); it != program->block()->end();
       ++it) {
    if ((*it)->isa<paddle::dialect::DataOp>()) {
      candidate_values.push_back((*it)->result(0));
    }
    if ((*it)->isa<cinn::dialect::GroupOp>()) {
      auto group_op = (*it)->dyn_cast<cinn::dialect::GroupOp>();
      for (auto* op : group_op.ops()) {
        if (op->isa<paddle::dialect::ExpOp>()) {
          candidate_values.push_back(op->result(0));
        }

        if (op->isa<paddle::dialect::SubtractOp>()) {
          candidate_values.push_back(op->result(0));
        }
      }
    }
  }

  std::vector<std::vector<pir::shape::SymbolicDimOp>> sym_vecs;

  for (auto value : candidate_values) {
    sym_vecs.push_back(
        shape_analysis->GetOrCreateSymbolicDimsForRankedValue(value));
  }

  sym_dim_mgr.MapSymbolicDimEqual(sym_vecs[1][0], sym_vecs[2][0]);
  sym_dim_mgr.MapSymbolicDimEqual(sym_vecs[1][0], sym_vecs[3][0]);

  std::cout << sym_dim_mgr.IsSymbolicDimEqual(sym_vecs[1][0], sym_vecs[2][0])
            << std::endl;
  std::cout << sym_vecs[1][0].GetSymName() << std::endl;
  std::cout << sym_vecs[1][1].GetSymName() << std::endl;
  std::cout << sym_vecs[2][0].GetSymName() << std::endl;
  std::cout << sym_vecs[2][1].GetSymName() << std::endl;

  std::cout << sym_vecs[3][0].GetSymName() << std::endl;
  std::cout << sym_vecs[3][1].GetSymName() << std::endl;

  return shape_analysis;
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

  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, *program);

  std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis =
      CreateShapeAnalysis(program);

  std::vector<pir::Value> test_values;

  for (auto it = program->block()->begin(); it != program->block()->end();
       ++it) {
    if ((*it)->isa<paddle::dialect::DataOp>()) {
      test_values.push_back((*it)->result(0));
    }
    if ((*it)->isa<cinn::dialect::GroupOp>()) {
      // GetOpList and Call cinn CodeGen
      auto group_op = (*it)->dyn_cast<cinn::dialect::GroupOp>();
      for (auto* op : group_op.ops()) {
        if (op->isa<paddle::dialect::ExpOp>()) {
          test_values.push_back(op->result(0));
        }
        if (op->isa<paddle::dialect::SubtractOp>()) {
          test_values.push_back(op->result(0));
        }
      }
      // auto sym_vec0 =
      // shape_analysis->GetOrCreateSymbolicDimsForRankedValue(test_values[0]);
      auto sym_vec1 =
          shape_analysis->GetOrCreateSymbolicDimsForRankedValue(test_values[1]);
      auto sym_vec2 =
          shape_analysis->GetOrCreateSymbolicDimsForRankedValue(test_values[2]);
      auto sym_vec3 =
          shape_analysis->GetOrCreateSymbolicDimsForRankedValue(test_values[3]);
      VLOG(1) << "%1 %2 IsShapeEqual:"
              << shape_analysis->IsShapeEqual(test_values[1], test_values[2]);
      VLOG(1) << "%1 %3 IsShapeEqual: "
              << shape_analysis->IsShapeEqual(test_values[1], test_values[3]);
      VLOG(1) << "%2 %3 IsShapeEqual: "
              << shape_analysis->IsShapeEqual(test_values[2], test_values[3]);
      VLOG(1) << sym_vec1[0].GetSymName() << " == " << sym_vec2[0].GetSymName()
              << " IsSymbolicDimEqual: "
              << shape_analysis->symbolicDimMgr().IsSymbolicDimEqual(
                     sym_vec1[0], sym_vec2[0]);
      VLOG(1) << sym_vec1[0].GetSymName() << " == " << sym_vec3[0].GetSymName()
              << " IsSymbolicDimEqual: "
              << shape_analysis->symbolicDimMgr().IsSymbolicDimEqual(
                     sym_vec1[0], sym_vec3[0]);
      VLOG(1) << sym_vec1[1].GetSymName() << " == " << sym_vec2[1].GetSymName()
              << " IsSymbolicDimEqual: "
              << shape_analysis->symbolicDimMgr().IsSymbolicDimEqual(
                     sym_vec1[1], sym_vec2[1]);
      VLOG(1) << sym_vec2[0].GetSymName() << " == " << sym_vec3[0].GetSymName()
              << " IsSymbolicDimEqual: "
              << shape_analysis->symbolicDimMgr().IsSymbolicDimEqual(
                     sym_vec2[0], sym_vec3[0]);
      VLOG(1) << sym_vec2[0].GetSymName() << " == " << sym_vec3[1].GetSymName()
              << " IsSymbolicDimEqual: "
              << shape_analysis->symbolicDimMgr().IsSymbolicDimEqual(
                     sym_vec2[0], sym_vec2[1]);

      // op fusion
      auto op_fusion = cinn::dialect::ir::OpFusionPassInternal(
          GetOpListNotIncludeYield(group_op.ops()), shape_analysis);

      // fusion merge
      auto group_list = cinn::dialect::ir::GeneralFusionMergePassInternal(
          op_fusion, shape_analysis);

      PADDLE_ENFORCE_EQ(group_list.size(),
                        1u,
                        phi::errors::Unimplemented(
                            "Only support one group after group fusion"));
      for (auto group : group_list) {
        auto ir_compiler = std::make_shared<cinn::hlir::framework::PirCompiler>(
            *program, target, scope);
        hlir::framework::PirCompilerManager::Instance().insert(ir_compiler);
        group->shape_analysis = shape_analysis;
        auto fn_ptr_res = ir_compiler->BuildCUDAJITInfo({group});
        std::unordered_map<std::string, ::pir::Attribute> op_attrs{
            {cinn::dialect::JitKernelOp::kAttrName,
             cinn::dialect::CUDAJITInfoAttribute::get(ctx, fn_ptr_res[0])},
        };

        // Generate jit kernel op input and output
        auto vec_ins = GetBlockOutsideInput(group->ops);

        std::vector<pir::Value> vec_new_ins;
        for (size_t i = 0; i < vec_ins.size(); ++i) {
          vec_new_ins.push_back(value_map.at(vec_ins[i]));
        }

        // using yield op to sort
        std::unordered_map<::pir::Value, size_t> value2id;
        auto yeild_op = group_op.ops().back();
        for (size_t i = 0; i < yeild_op->num_operands(); ++i) {
          value2id[yeild_op->operand_source(i)] = i;
        }

        std::unordered_map<size_t, size_t> codegen2orig;

        std::vector<pir::Type> vec_types;
        for (size_t i = 0; i < group->output_values.size(); ++i) {
          vec_types.push_back(group->output_values[i].type());
          codegen2orig[value2id.at(group->output_values[i])] = i;
        }

        ::pir::Operation* cinn_op =
            ::pir::Operation::Create(vec_new_ins, op_attrs, vec_types, op_info);

        for (size_t i = 0; i < group_op.num_results(); ++i) {
          value_map[group_op.result(i)] = cinn_op->result(codegen2orig.at(i));
        }

        ir_program->block()->push_back(cinn_op);
      }

    } else {
      std::vector<pir::Value> vec_ins;

      for (size_t i = 0; i < (*it)->num_operands(); ++i) {
        if ((*it)->operand_source(i)) {
          vec_ins.push_back(value_map.at((*it)->operand_source(i)));
        } else {
          vec_ins.push_back((*it)->operand_source(i));
        }
      }

      std::vector<pir::Type> vec_types;
      for (size_t i = 0; i < (*it)->num_results(); ++i) {
        vec_types.push_back((*it)->result(i).type());
      }

      ::pir::OpInfo info1 = ctx->GetRegisteredOpInfo((*it)->name());
      ::pir::Operation* op = ::pir::Operation::Create(
          vec_ins, (*it)->attributes(), vec_types, info1);

      ir_program->block()->push_back(op);
      for (size_t i = 0; i < (*it)->num_results(); ++i) {
        value_map[(*it)->result(i)] = op->result(i);
      }
    }
  }
  return ir_program;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
