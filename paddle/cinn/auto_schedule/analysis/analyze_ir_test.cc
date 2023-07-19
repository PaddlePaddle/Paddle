// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/analysis/analyze_ir.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "paddle/cinn/common/context.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/poly/stage.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

TEST(AnalyzeIr, AnalyzeScheduleBlockReadWriteBuffer_SimpleAssign) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ir::Expr M(32);
  ir::Expr N(32);

  lang::Placeholder<float> A("A", {M, N});
  ir::Tensor B = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  poly::StageMap stages = poly::CreateStages({A, B});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      "SimpleAssign", stages, {A, B}, {}, {}, nullptr, target, true);

  ASSERT_FALSE(funcs.empty());
  ir::Expr ast_expr = funcs[0]->body;

  VLOG(6) << "Analyzing for Expr:";
  VLOG(6) << ast_expr;

  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  std::vector<ir::Expr> all_block_realizes = ir_sch.GetAllBlocks();
  ASSERT_EQ(all_block_realizes.size(), 1UL);

  ir::ScheduleBlockRealize* sche_block_realize =
      all_block_realizes[0].As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sche_block =
      sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
  AnalyzeScheduleBlockReadWriteBuffer(sche_block);

  /*
   * the sche_block_realize will be:
   * ScheduleBlock(B)
   * {
   *   i0, i1 = axis.bind(i, j)
   *   read_buffers(_A[i0(undefined:undefined), i1(undefined:undefined)])
   *   write_buffers(_B[i0(undefined:undefined), i1(undefined:undefined)])
   *   B[i0, i1] = A[i0, i1]
   * }
   */

  VLOG(6) << "ScheduleBlockRealize: ";
  VLOG(6) << all_block_realizes[0];

  ASSERT_EQ(sche_block->read_buffers.size(), 1UL);

  std::stringstream read_ss;
  read_ss << sche_block->read_buffers[0];
  ASSERT_EQ(read_ss.str(), "_A[i0(0:32), i1(0:32)]");

  ASSERT_EQ(sche_block->write_buffers.size(), 1UL);
  std::stringstream write_ss;
  write_ss << sche_block->write_buffers[0];
  ASSERT_EQ(write_ss.str(), "_B[i0(0:32), i1(0:32)]");
}

TEST(AnalyzeIr, AnalyzeScheduleBlockReadWriteBuffer_AddDiffShape) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ir::Expr M(32);
  ir::Expr N(128);

  lang::Placeholder<float> A("A", {M});
  lang::Placeholder<float> B("B", {N});

  ir::Tensor C = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i) + B(j); }, "C");

  poly::StageMap stages = poly::CreateStages({C});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      "AddDiffShape", stages, {C}, {}, {}, nullptr, target, true);

  ir::Expr ast_expr = funcs[0]->body;
  VLOG(6) << "Expr before MultiLevelTiling: ";
  VLOG(6) << ast_expr;

  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  std::vector<ir::Expr> all_block_realizes = ir_sch.GetAllBlocks();
  ASSERT_EQ(all_block_realizes.size(), 1UL);

  ir::ScheduleBlockRealize* sche_block_realize =
      all_block_realizes[0].As<ir::ScheduleBlockRealize>();
  ir::ScheduleBlock* sche_block =
      sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
  AnalyzeScheduleBlockReadWriteBuffer(sche_block);

  VLOG(6) << "ScheduleBlockRealize: ";
  VLOG(6) << all_block_realizes[0];
  ASSERT_EQ(sche_block->read_buffers.size(), 2UL);
  std::vector<std::string> expect_read = {"_A[i0(0:32)]", "_B[i1(0:128)]"};

  ASSERT_EQ(sche_block->read_buffers.size(), expect_read.size());
  for (size_t i = 0; i < expect_read.size(); ++i) {
    std::stringstream read_ss;
    read_ss << sche_block->read_buffers[i];
    ASSERT_EQ(read_ss.str(), expect_read[i]);
  }

  ASSERT_EQ(sche_block->write_buffers.size(), 1UL);
  std::stringstream write_ss;
  write_ss << sche_block->write_buffers[0];
  ASSERT_EQ(write_ss.str(), "_C[i0(0:32), i1(0:128)]");
}

TEST(AnalyzeIr, ContainsNodeType) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif

  ir::Expr M(32);
  ir::Expr N(32);

  lang::Placeholder<float> A("A", {M, N});
  ir::Tensor B = lang::Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  poly::StageMap stages = poly::CreateStages({A, B});
  std::vector<ir::LoweredFunc> funcs = lang::LowerVec(
      "SimpleAssign", stages, {A, B}, {}, {}, nullptr, target, true);

  ASSERT_FALSE(funcs.empty());
  ir::Expr ast_expr = funcs[0]->body;

  VLOG(6) << "Analyzing for Expr:";
  VLOG(6) << ast_expr;

  ASSERT_TRUE(
      ContainsNodeType(ast_expr, {ir::IrNodeTy::Load, ir::IrNodeTy::Store}));
  ASSERT_TRUE(ContainsNodeType(ast_expr,
                               {ir::IrNodeTy::Load, ir::IrNodeTy::IfThenElse}));
  ASSERT_FALSE(ContainsNodeType(ast_expr,
                                {ir::IrNodeTy::IfThenElse, ir::IrNodeTy::Sum}));
}

}  // namespace auto_schedule
}  // namespace cinn
