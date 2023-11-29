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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sstream>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/adt/map_expr_ctx.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

PD_DECLARE_bool(cinn_enable_map_expr);
PD_DECLARE_bool(cinn_enable_map_expr_dynamic_shape);

std::vector<pir::OpResult> BuildInput(
    ::pir::Builder* builder,
    const std::vector<std::vector<int64_t>>& vec_shapes) {
  std::vector<pir::OpResult> vec_res;
  for (size_t i = 0; i < vec_shapes.size(); ++i) {
    auto op = builder->Build<paddle::dialect::FullOp>(
        vec_shapes[i], 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

    vec_res.push_back(op.result(0));
  }

  return vec_res;
}

class MockShapeConstraintIRAnalysis : public pir::ShapeConstraintIRAnalysis {
 public:
  explicit MockShapeConstraintIRAnalysis(
      std::unique_ptr<pir::Program>&& program)
      : pir::ShapeConstraintIRAnalysis(program->module_op()),
        program_(std::move(program)) {}

  explicit MockShapeConstraintIRAnalysis(pir::IrContext* ctx)
      : MockShapeConstraintIRAnalysis(std::make_unique<pir::Program>(ctx)) {}

  MockShapeConstraintIRAnalysis(MockShapeConstraintIRAnalysis&& other) = delete;
  MockShapeConstraintIRAnalysis(const MockShapeConstraintIRAnalysis& other) =
      delete;

  const std::unordered_map<pir::Value, std::vector<pir::shape::SymbolicDimOp>>&
  value_to_sym_dims() const {
    return value_to_sym_dims_;
  }

  std::unordered_map<pir::Value, std::vector<pir::shape::SymbolicDimOp>>*
  mut_value_to_sym_dims() {
    return &value_to_sym_dims_;
  }

 private:
  std::unique_ptr<pir::Program> program_;
};

std::shared_ptr<pir::ShapeConstraintIRAnalysis> CreateShapeAnalysis(
    const pir::Program* program) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();

  auto shape_analysis = std::make_shared<MockShapeConstraintIRAnalysis>(ctx);
  pir::SymbolicDimMgr& sym_dim_mgr = shape_analysis->symbolicDimMgr();

  std::vector<std::vector<pir::shape::SymbolicDimOp>> datas_sym_vec{};

  for (auto it = program->block()->begin(); it != program->block()->end();
       ++it) {
    if (it->isa<cinn::dialect::GroupOp>()) {
      auto group_op = it->dyn_cast<cinn::dialect::GroupOp>();
      for (auto* op : group_op.ops()) {
        if (op->isa<paddle::dialect::ExpOp>()) {
          datas_sym_vec.emplace_back(
              shape_analysis->GetOrCreateSymbolicDimsForRankedValue(
                  op->result(0)));
        }

        if (op->isa<paddle::dialect::SubtractOp>()) {
          datas_sym_vec.emplace_back(
              shape_analysis->GetOrCreateSymbolicDimsForRankedValue(
                  op->result(0)));
        }
      }
    }
    if (it->isa<paddle::dialect::DataOp>()) {
      auto op = it->dyn_cast<paddle::dialect::DataOp>();
      datas_sym_vec.emplace_back(
          shape_analysis->GetOrCreateSymbolicDimsForRankedValue(op->result(0)));
    }
  }

  for (std::size_t i = 1; i < datas_sym_vec.size(); ++i) {
    sym_dim_mgr.MapSymbolicDimEqual(datas_sym_vec.at(0)[0],
                                    datas_sym_vec.at(i)[0]);
    sym_dim_mgr.MapSymbolicDimEqual(datas_sym_vec.at(0)[1],
                                    datas_sym_vec.at(i)[1]);
  }

  CHECK_NOTNULL(shape_analysis.get());
  return shape_analysis;
}

TEST(MapExpr, ElementWise_Fusion_0) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program_base(ctx);
  ::pir::Builder builder_base = ::pir::Builder(ctx, program_base.block());

  auto inputs = BuildInput(&builder_base, {{-1, 1}, {-1, 1}});

  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  builder.Build<paddle::dialect::SubtractOp>(
      inputs[0], builder.Build<paddle::dialect::ExpOp>(inputs[1]).result(0));

  std::vector<pir::Operation*> vec_op;
  for (auto& op : *program.block()) {
    vec_op.push_back(&op);
  }

  auto res = cinn::dialect::ir::OpFusionPassInternal(vec_op);
  ASSERT_EQ(res.size(), 1u);
  ASSERT_EQ(res[0]->ops.size(), program.block()->size());

  auto group_list = cinn::dialect::ir::GeneralFusionMergePassInternal(res);
  ASSERT_EQ(group_list.size(), 1u);

  FLAGS_cinn_enable_map_expr = true;
  FLAGS_cinn_enable_map_expr_dynamic_shape = true;

  for (auto group : group_list) {
    group->shape_analysis = CreateShapeAnalysis(&program);
    cinn::adt::TryGenerateMapExprFromGroup(group);
    VLOG(1) << "MapExpr: "
            << cinn::adt::ToTxtString(group->map_expr_ctx().map_expr(), "");
  }
}
