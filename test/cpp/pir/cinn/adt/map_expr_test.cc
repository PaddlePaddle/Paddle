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
#include "paddle/fluid/pir/transforms/infer_symbolic_shape_pass.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"
#include "paddle/pir/pass/pass_manager.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

PD_DECLARE_bool(cinn_enable_map_expr);
PD_DECLARE_bool(cinn_enable_map_expr_dynamic_shape);
PD_DECLARE_bool(cinn_enable_map_expr_index_detail);

TEST(MapExpr, ElementWise_Fusion_0) {
  cinn::adt::UniqueId::ResetSeqNumber(0);
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ::pir::Program program(ctx);
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  phi::DDim dims_D_2 = {-1, 1};
  pir::OpResult value1 =
      test::CreateDenseTensorOp(ctx, dims_D_2, {"op1_attr"}, {"op1_name"})
          ->result(0);
  pir::OpResult value2 =
      test::CreateDenseTensorOp(ctx, dims_D_2, {"op2_attr"}, {"op2_name"})
          ->result(0);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  builder.Build<paddle::dialect::SubtractOp>(
      value1, builder.Build<paddle::dialect::ExpOp>(value2).result(0));

  ::pir::PassManager pass_manager(ctx);
  auto shape_analysis =
      std::make_shared<pir::MockShapeConstraintIRAnalysis>(ctx);
  pass_manager.AddPass(::pir::CreateInferSymbolicShapePass(shape_analysis));
  pass_manager.Run(&program);

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
  FLAGS_cinn_enable_map_expr_index_detail = true;

  auto group = group_list.at(0);
  group->shape_analysis = shape_analysis;
  cinn::adt::TryGenerateMapExprFromGroup(group);
  std::string map_expr_str =
      cinn::adt::ToTxtString(group->map_expr_ctx().map_expr(), "MapExprTest");
  std::string target_str =
      "\nMapExprTest(t_var_2, t_var_1) {\n"
      "  AnchoredMapStmt(t_var_0) {\n"
      "    MapStmt([i_59, i_60]) {\n"
      "      exp(\n"
      "          &t_var[IndexDot([BI(i_59, sym_17), 0], [sym_17, 1])], \n"
      "          t_var_1[IndexDot([BI(i_59, sym_17), 0], [sym_17, 1])]);\n"
      "      subtract(\n"
      "          &t_var_0[IndexDot([i_59, i_60], [sym_17, 1])], \n"
      "          t_var_2[IndexDot([BI(i_59, sym_17), 0], [sym_17, 1])], \n"
      "          t_var[IndexDot([BI(i_59, sym_17), 0], [sym_17, 1])]);\n"
      "    }\n"
      "  }\n"
      "}\n";
  ASSERT_EQ(map_expr_str, target_str);
}
