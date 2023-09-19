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

#include "paddle/cinn/ir/schedule/schedule_desc.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace ir {

// Return lowerd ir AST for example functions used in this test
std::vector<ir::LoweredFunc> LowerCompute(
    const std::vector<int>& shape,
    const Target& target,
    bool need_c = false,
    const std::string& operation = "elementwise-copy") {
  CHECK(shape.size() == 2 || shape.size() == 3) << "shape should be 2 or 3";
  std::vector<Expr> domain;
  for (auto i = 0; i < shape.size(); ++i) {
    domain.emplace_back(shape[i]);
  }

  Placeholder<float> A("A", domain);
  ir::Tensor B, C;

  if (operation == "elementwise-copy") {
    if (domain.size() == 2) {
      B = Compute(
          domain, [&A](Var i, Var j) { return A(i, j); }, "B");
      C = Compute(
          domain, [&B](Var i, Var j) { return B(i, j); }, "C");
    } else {
      B = Compute(
          domain, [&A](Var i, Var j, Var k) { return A(i, j, k); }, "B");
      C = Compute(
          domain, [&B](Var i, Var j, Var k) { return B(i, j, k); }, "C");
    }
  }

  if (operation == "elementwise-add_const") {
    if (domain.size() == 2) {
      B = Compute(
          domain, [&A](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
      C = Compute(
          domain, [&B](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");
    } else {
      B = Compute(
          domain,
          [&A](Var i, Var j, Var k) { return A(i, j, k) * Expr(2.f); },
          "B");
      C = Compute(
          domain,
          [&B](Var i, Var j, Var k) { return B(i, j, k) + Expr(1.f); },
          "C");
    }
  }

  if (need_c) {
    return cinn::lang::LowerVec("test_func",
                                CreateStages({A, B, C}),
                                {A, C},
                                {},
                                {},
                                nullptr,
                                target,
                                true);
  }

  return cinn::lang::LowerVec(
      "test_func", CreateStages({A, B}), {A, B}, {}, {}, nullptr, target, true);
}

// Create a new IRSchedule with copied ir::LoweredFunc AST
IRSchedule MakeIRSchedule(const std::vector<ir::LoweredFunc>& lowered_funcs) {
  std::vector<Expr> exprs;
  for (auto&& func : lowered_funcs) {
    exprs.emplace_back(ir::ir_utils::IRCopy(func->body));
  }
  return ir::IRSchedule(ir::ModuleExpr(exprs));
}

// Generate source code with transformed ModuleExpr
std::string SourceCodeGen(const ModuleExpr& module_expr,
                          const std::vector<ir::LoweredFunc>& lowered_funcs,
                          const Target& target) {
  auto exprs = module_expr.GetExprs();
  CHECK_EQ(exprs.size(), lowered_funcs.size()) << "size of func is not euqal";
  std::vector<ir::LoweredFunc> updated_funcs =
      ir::ir_utils::IRCopy(lowered_funcs);
  Module::Builder builder("test_module", target);
  for (auto i = 0; i < lowered_funcs.size(); ++i) {
    updated_funcs[i]->body = ir::ir_utils::IRCopy(exprs.at(i));
    builder.AddFunction(updated_funcs[i]);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  return codegen.Compile(module, CodeGenC::OutputKind::CImpl);
}

class TestScheduleDesc : public ::testing::Test {
 public:
  Target target = common::DefaultHostTarget();
  std::vector<ir::LoweredFunc> lowered_funcs;
  ScheduleDesc trace;
  void SetUp() override { Context::Global().ResetNameId(); }

  void CheckTracingOutputs(const std::vector<Expr>& base,
                           const ScheduleDesc& trace_desc) {
    Context::Global().ResetNameId();
    ir::IRSchedule replay_sch = MakeIRSchedule(lowered_funcs);
    auto traced_outputs =
        ScheduleDesc::ReplayWithProto(trace_desc.ToProto(), &replay_sch);
    ASSERT_EQ(base.size(), traced_outputs.size());
    for (auto i = 0; i < base.size(); ++i) {
      ASSERT_EQ(utils::GetStreamCnt(base.at(i)),
                utils::GetStreamCnt(traced_outputs.at(i)));
    }
  }

  void CheckReplayResult(const ir::IRSchedule& ir_sch,
                         const ScheduleDesc& trace_desc) {
    Context::Global().ResetNameId();
    ir::IRSchedule replay_sch = MakeIRSchedule(lowered_funcs);
    trace_desc.Replay(&replay_sch);

    // check the equality of module expr between original schedule
    // and the schedule generated by replaying with tracing ScheduleDesc
    auto lhs_exprs = ir_sch.GetModule().GetExprs();
    auto rhs_exprs = replay_sch.GetModule().GetExprs();
    ASSERT_EQ(lhs_exprs.size(), rhs_exprs.size());
    for (auto i = 0; i < lhs_exprs.size(); ++i) {
      ASSERT_EQ(utils::GetStreamCnt(lhs_exprs.at(i)),
                utils::GetStreamCnt(rhs_exprs.at(i)));
    }

    // check the equality of source code between them
    ASSERT_EQ(
        utils::Trim(SourceCodeGen(ir_sch.GetModule(), lowered_funcs, target)),
        utils::Trim(
            SourceCodeGen(replay_sch.GetModule(), lowered_funcs, target)));
  }
};

TEST_F(TestScheduleDesc, Append_Replay) {
  lowered_funcs = LowerCompute({32, 32}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto fused = ir_sch.Fuse("B", {0, 1});
  trace.Append(ScheduleDesc::Step("FuseWithName",
                                  {},
                                  {{"block_name", std::string("B")},
                                   {"loops_index", std::vector<int>({0, 1})}},
                                  {fused}));
  auto sample = ir_sch.SamplePerfectTile(fused, 2, 1, {4, -1});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop", std::vector<Expr>({fused})}},
                                  {{"n", 2},
                                   {"max_innermost_factor", 1},
                                   {"decision", std::vector<int>{4, -1}}},
                                  sample));
  auto splited = ir_sch.Split(fused, sample);
  trace.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({fused})}, {"factors", sample}},
      {},
      splited));

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  fused = ir_sch.Fuse(loops);
  trace.Append(ScheduleDesc::Step("Fuse", {{"loops", loops}}, {}, {fused}));
  sample = ir_sch.SamplePerfectTile(fused, 2, 1, {256, -1});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop", std::vector<Expr>({fused})}},
                                  {{"n", 2},
                                   {"max_innermost_factor", 1},
                                   {"decision", std::vector<int>{256, -1}}},
                                  sample));
  splited = ir_sch.Split(fused, sample);
  trace.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({fused})}, {"factors", sample}},
      {},
      splited));

  // check the equality of results between the ir_sch and replaying of trace
  CheckTracingOutputs(splited, trace);
  CheckReplayResult(ir_sch, trace);
  // check the equality of results between the ir_sch and replaying of its trace
  CheckTracingOutputs(splited, ir_sch.GetTraceDesc());
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

// Test cases with `StepKind` prefix are to check the correctness of their
// StepKindInfo register
TEST_F(TestScheduleDesc, StepKind_GetAllBlocks) {
  lowered_funcs = LowerCompute({32, 32}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto all_blocks = ir_sch.GetAllBlocks();
  trace.Append(ScheduleDesc::Step("GetAllBlocks", {}, {}, {all_blocks}));
  CheckTracingOutputs(all_blocks, trace);
  CheckTracingOutputs(all_blocks, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_GetChildBlocks) {
  lowered_funcs = LowerCompute({32, 32, 64}, target, true);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  auto loops = ir_sch.GetLoops("C");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("C")}}, loops));
  ir_sch.ComputeAt(block_b, loops[1]);
  trace.Append(ScheduleDesc::Step("ComputeAt",
                                  {{"block", std::vector<Expr>({block_b})},
                                   {"loop", std::vector<Expr>({loops[1]})}},
                                  {{"keep_unit_loops", false}},
                                  {}));
  loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  auto root_block = ir_sch.GetRootBlock(loops[1]);
  trace.Append(ScheduleDesc::Step("GetRootBlock",
                                  {{"expr", std::vector<Expr>({loops[1]})}},
                                  {},
                                  {root_block}));
  auto childblocks = ir_sch.GetChildBlocks(root_block);
  trace.Append(ScheduleDesc::Step("GetChildBlocks",
                                  {{"expr", std::vector<Expr>({root_block})}},
                                  {},
                                  childblocks));
  CheckTracingOutputs(childblocks, trace);
  CheckTracingOutputs(childblocks, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_GetLoops) {
  lowered_funcs = LowerCompute({32, 32}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  auto loops = ir_sch.GetLoops(block_b);
  trace.Append(ScheduleDesc::Step(
      "GetLoops", {{"block", std::vector<Expr>({block_b})}}, {}, loops));
  CheckTracingOutputs(loops, trace);
  CheckTracingOutputs(loops, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_GetLoopsWithName) {
  lowered_funcs = LowerCompute({32, 32}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  CheckTracingOutputs(loops, trace);
  CheckTracingOutputs(loops, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_GetBlock) {
  lowered_funcs = LowerCompute({32, 32, 32}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  CheckTracingOutputs({block_b}, trace);
  CheckTracingOutputs({block_b}, ir_sch.GetTraceDesc());
}
// TODO(SunNy820828449): fix in future, as fix split var name, this case some
// problem.
/*
TEST_F(TestScheduleDesc, StepKind_Split) {
  lowered_funcs                         = LowerCompute({32, 32, 32}, target);
  ir::IRSchedule ir_sch_split_base      = MakeIRSchedule(lowered_funcs);
  ir::IRSchedule ir_sch_split           = MakeIRSchedule(lowered_funcs);
  ir::IRSchedule ir_sch_split_with_name = MakeIRSchedule(lowered_funcs);

  // test split with inputs of Expr
  auto loops = ir_sch_split_base.GetLoops("B");
  trace.Append(ScheduleDesc::Step("GetLoopsWithName", {}, {{"block_name",
std::string("B")}}, loops)); auto sample =
ir_sch_split_base.SamplePerfectTile(loops.front(), 2, 1, {4, -1});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop",
std::vector<Expr>({loops.front()})}},
                                  {{"n", 2}, {"max_innermost_factor", 1},
{"decision", std::vector<int>{4, -1}}}, sample)); auto splited =
ir_sch_split_base.Split(loops.front(), sample); trace.Append(
      ScheduleDesc::Step("Split", {{"loop", std::vector<Expr>({loops.front()})},
{"factors", sample}}, {}, splited)); CheckTracingOutputs(splited, trace);
  CheckTracingOutputs(splited, ir_sch_split_base.GetTraceDesc());

  // test split with inputs of int
  loops   = ir_sch_split.GetLoops("B");
  splited = ir_sch_split.Split(loops.front(), {4, -1});
  CheckTracingOutputs(splited, trace);
  CheckTracingOutputs(splited, ir_sch_split.GetTraceDesc());

  // test split with block name and inputs of int
  splited = ir_sch_split_with_name.Split("B", 0, {4, -1});
  CheckTracingOutputs(splited, trace);
  CheckTracingOutputs(splited, ir_sch_split_with_name.GetTraceDesc());
}
*/
TEST_F(TestScheduleDesc, StepKind_Fuse) {
  lowered_funcs = LowerCompute({32, 32, 64}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  auto fused = ir_sch.Fuse(loops);
  trace.Append(ScheduleDesc::Step("Fuse", {{"loops", loops}}, {}, {fused}));
  CheckTracingOutputs({fused}, trace);
  CheckTracingOutputs({fused}, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_FuseWithName) {
  lowered_funcs = LowerCompute({32, 32, 64}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto fused = ir_sch.Fuse("B", {0, 1, 2});
  trace.Append(
      ScheduleDesc::Step("FuseWithName",
                         {},
                         {{"block_name", std::string("B")},
                          {"loops_index", std::vector<int>({0, 1, 2})}},
                         {fused}));
  CheckTracingOutputs({fused}, trace);
  CheckTracingOutputs({fused}, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_FuseWithBlock) {
  lowered_funcs = LowerCompute({32, 32, 64}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  auto fused = ir_sch.Fuse(block_b, {0, 1, 2});
  trace.Append(
      ScheduleDesc::Step("FuseWithBlock",
                         {{"block", std::vector<Expr>({block_b})}},
                         {{"loops_index", std::vector<int>({0, 1, 2})}},
                         {fused}));
  CheckTracingOutputs({fused}, trace);
  CheckTracingOutputs({fused}, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_ComputeAt) {
  lowered_funcs = LowerCompute({32, 32, 64}, target, true);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  auto loops = ir_sch.GetLoops("C");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("C")}}, loops));
  ir_sch.ComputeAt(block_b, loops[1]);
  trace.Append(ScheduleDesc::Step("ComputeAt",
                                  {{"block", std::vector<Expr>({block_b})},
                                   {"loop", std::vector<Expr>({loops[1]})}},
                                  {{"keep_unit_loops", false}},
                                  {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_SimpleComputeAt) {
  lowered_funcs = LowerCompute({32, 32, 64}, target, true);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  auto loops = ir_sch.GetLoops("C");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("C")}}, loops));
  ir_sch.SimpleComputeAt(block_b, loops[2]);
  trace.Append(ScheduleDesc::Step("SimpleComputeAt",
                                  {{"block", std::vector<Expr>({block_b})},
                                   {"loop", std::vector<Expr>({loops[2]})}},
                                  {{"keep_unit_loops", false}},
                                  {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_ReverseComputeAt) {
  lowered_funcs = LowerCompute({32, 32, 64}, target, true);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_c = ir_sch.GetBlock("C");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("C")}}, {block_c}));
  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  ir_sch.ReverseComputeAt(block_c, loops[1]);
  trace.Append(ScheduleDesc::Step("ReverseComputeAt",
                                  {{"block", std::vector<Expr>({block_c})},
                                   {"loop", std::vector<Expr>({loops[1]})}},
                                  {{"keep_unit_loops", false}},
                                  {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_GetRootBlock) {
  lowered_funcs = LowerCompute({32, 64}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  auto root_b = ir_sch.GetRootBlock(loops[1]);
  trace.Append(ScheduleDesc::Step(
      "GetRootBlock", {{"expr", std::vector<Expr>({loops[1]})}}, {}, {root_b}));
  CheckTracingOutputs({root_b}, trace);
  CheckTracingOutputs({root_b}, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_CacheRead) {
  lowered_funcs =
      LowerCompute({32, 64}, target, false, "elementwise-add_const");
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  auto a_cache = ir_sch.CacheRead(block_b, 0, "local");
  trace.Append(ScheduleDesc::Step(
      "CacheRead",
      {{"block", std::vector<Expr>({block_b})}},
      {{"read_buffer_index", 0}, {"memory_type", std::string("local")}},
      {a_cache}));
  CheckTracingOutputs({a_cache}, trace);
  CheckTracingOutputs({a_cache}, ir_sch.GetTraceDesc());
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_CacheWrite) {
  lowered_funcs =
      LowerCompute({32, 64}, target, false, "elementwise-add_const");
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  trace.Append(ScheduleDesc::Step(
      "CacheWrite",
      {{"block", std::vector<Expr>({block_b})}},
      {{"write_buffer_index", 0}, {"memory_type", std::string("local")}},
      {b_cache}));
  CheckTracingOutputs({b_cache}, trace);
  CheckTracingOutputs({b_cache}, ir_sch.GetTraceDesc());
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_SyncThreads) {
  lowered_funcs = LowerCompute({64, 32}, target, true, "elementwise-add_const");
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  trace.Append(ScheduleDesc::Step(
      "CacheWrite",
      {{"block", std::vector<Expr>({block_b})}},
      {{"write_buffer_index", 0}, {"memory_type", std::string("local")}},
      {b_cache}));
  auto block_c = ir_sch.GetBlock("C");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("C")}}, {block_c}));
  auto c_cache = ir_sch.CacheWrite(block_c, 0, "local");
  trace.Append(ScheduleDesc::Step(
      "CacheWrite",
      {{"block", std::vector<Expr>({block_c})}},
      {{"write_buffer_index", 0}, {"memory_type", std::string("local")}},
      {c_cache}));
  block_c = ir_sch.GetBlock("C");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("C")}}, {block_c}));
  ir_sch.SyncThreads(block_c, false);
  trace.Append(ScheduleDesc::Step("SyncThreads",
                                  {{"ir_node", std::vector<Expr>({block_c})}},
                                  {{"after_node", false}},
                                  {}));
  block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.SyncThreads(block_b);
  trace.Append(ScheduleDesc::Step("SyncThreads",
                                  {{"ir_node", std::vector<Expr>({block_b})}},
                                  {{"after_node", true}},
                                  {}));

  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_SetBuffer) {
  lowered_funcs =
      LowerCompute({32, 64}, target, false, "elementwise-add_const");
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.SetBuffer(block_b, "shared", true);
  trace.Append(ScheduleDesc::Step(
      "SetBuffer",
      {{"block", std::vector<Expr>({block_b})}},
      {{"memory_type", std::string("shared")}, {"fixed", true}},
      {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_Reorder) {
  lowered_funcs = LowerCompute({32, 64, 12}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  auto sample = ir_sch.SamplePerfectTile(loops[0], 2, 1, {-1, 4});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop", std::vector<Expr>({loops[0]})}},
                                  {{"n", 2},
                                   {"max_innermost_factor", 1},
                                   {"decision", std::vector<int>{-1, 4}}},
                                  sample));
  auto splited = ir_sch.Split(loops[0], sample);
  trace.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({loops[0]})}, {"factors", sample}},
      {},
      splited));

  loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  sample = ir_sch.SamplePerfectTile(loops[2], 2, 1, {-1, 2});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop", std::vector<Expr>({loops[2]})}},
                                  {{"n", 2},
                                   {"max_innermost_factor", 1},
                                   {"decision", std::vector<int>{-1, 2}}},
                                  sample));
  splited = ir_sch.Split(loops[2], sample);
  trace.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({loops[2]})}, {"factors", sample}},
      {},
      splited));

  loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  Expr ret = ir_sch.Reorder({loops[4], loops[0]});
  trace.Append(
      ScheduleDesc::Step("Reorder",
                         {{"loops", std::vector<Expr>({loops[4], loops[0]})}},
                         {},
                         {ret}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_ReorderWithBlock) {
  lowered_funcs = LowerCompute({32, 32, 64}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);
  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  auto sample = ir_sch.SamplePerfectTile(loops[0], 2, 1, {-1, 4});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop", std::vector<Expr>({loops[0]})}},
                                  {{"n", 2},
                                   {"max_innermost_factor", 1},
                                   {"decision", std::vector<int>{-1, 4}}},
                                  sample));
  auto splited = ir_sch.Split(loops[0], sample);
  trace.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({loops[0]})}, {"factors", sample}},
      {},
      splited));

  loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  sample = ir_sch.SamplePerfectTile(loops[2], 2, 1, {-1, 2});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop", std::vector<Expr>({loops[2]})}},
                                  {{"n", 2},
                                   {"max_innermost_factor", 1},
                                   {"decision", std::vector<int>{-1, 2}}},
                                  sample));
  splited = ir_sch.Split(loops[2], sample);
  trace.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({loops[2]})}, {"factors", sample}},
      {},
      splited));

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  Expr ret = ir_sch.Reorder("B", {2, 3, 1, 4, 0});
  trace.Append(
      ScheduleDesc::Step("ReorderWithBlock",
                         {{"block", std::vector<Expr>({block_b})}},
                         {{"loops_index", std::vector<int>({2, 3, 1, 4, 0})}},
                         {ret}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_ReorderWithName) {
  lowered_funcs = LowerCompute({32, 32, 64}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  auto sample = ir_sch.SamplePerfectTile(loops[0], 2, 1, {-1, 4});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop", std::vector<Expr>({loops[0]})}},
                                  {{"n", 2},
                                   {"max_innermost_factor", 1},
                                   {"decision", std::vector<int>{-1, 4}}},
                                  sample));
  auto splited = ir_sch.Split(loops[0], sample);
  trace.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({loops[0]})}, {"factors", sample}},
      {},
      splited));

  loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  sample = ir_sch.SamplePerfectTile(loops[2], 2, 1, {-1, 2});
  trace.Append(ScheduleDesc::Step("SamplePerfectTile",
                                  {{"loop", std::vector<Expr>({loops[2]})}},
                                  {{"n", 2},
                                   {"max_innermost_factor", 1},
                                   {"decision", std::vector<int>{-1, 2}}},
                                  sample));
  splited = ir_sch.Split(loops[2], sample);
  trace.Append(ScheduleDesc::Step(
      "Split",
      {{"loop", std::vector<Expr>({loops[2]})}, {"factors", sample}},
      {},
      splited));

  Expr ret = ir_sch.Reorder("B", {4, 2, 3, 1, 0});
  trace.Append(
      ScheduleDesc::Step("ReorderWithName",
                         {},
                         {{"block_name", std::string("B")},
                          {"loops_index", std::vector<int>({4, 2, 3, 1, 0})}},
                         {ret}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_Parallel) {
  lowered_funcs = LowerCompute({32, 64}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  ir_sch.Parallel(loops[0]);
  trace.Append(ScheduleDesc::Step(
      "Parallel", {{"loop", std::vector<Expr>({loops[0]})}}, {}, {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_Vectorize) {
  lowered_funcs = LowerCompute({32, 64}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  ir_sch.Vectorize(loops[1], 16);
  trace.Append(ScheduleDesc::Step("Vectorize",
                                  {{"loop", std::vector<Expr>({loops[1]})}},
                                  {{"factor", 16}},
                                  {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_Unroll) {
  lowered_funcs = LowerCompute({32, 2}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  ir_sch.Unroll(loops[1]);
  trace.Append(ScheduleDesc::Step(
      "Unroll", {{"loop", std::vector<Expr>({loops[1]})}}, {}, {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_ComputeInline) {
  lowered_funcs =
      LowerCompute({32, 32, 32}, target, true, "elementwise-add_const");
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.ComputeInline(block_b);
  trace.Append(
      ScheduleDesc::Step("ComputeInline",
                         {{"schedule_block", std::vector<Expr>({block_b})}},
                         {},
                         {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_ReverseComputeInline) {
  lowered_funcs =
      LowerCompute({32, 32, 32}, target, true, "elementwise-add_const");
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);
  auto block_c = ir_sch.GetBlock("C");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("C")}}, {block_c}));
  ir_sch.ReverseComputeInline(block_c);
  trace.Append(
      ScheduleDesc::Step("ReverseComputeInline",
                         {{"schedule_block", std::vector<Expr>({block_c})}},
                         {},
                         {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_Bind) {
  lowered_funcs = LowerCompute({32, 128}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  ir_sch.Bind(loops[0], "blockIdx.x");
  trace.Append(ScheduleDesc::Step("Bind",
                                  {{"loop", std::vector<Expr>({loops[0]})}},
                                  {{"thread_axis", std::string("blockIdx.x")}},
                                  {}));
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_Rfactor) {
  Expr M(32);
  Expr N(2);
  Expr K(16);

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});
  Var k(16, "k0");
  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return lang::ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  lowered_funcs = cinn::lang::LowerVec("test_rfactor",
                                       CreateStages({A, B, C}),
                                       {A, B, C},
                                       {},
                                       {},
                                       nullptr,
                                       target,
                                       true);

  cinn::common::Context::Global().ResetNameId();
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);
  cinn::common::Context::Global().ResetNameId();

  auto loops = ir_sch.GetLoops("C");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("C")}}, loops));
  auto new_rf_tensor = ir_sch.Rfactor(loops[2], 0);
  trace.Append(ScheduleDesc::Step("Rfactor",
                                  {{"rf_loop", std::vector<Expr>({loops[2]})}},
                                  {{"rf_axis", 0}},
                                  {new_rf_tensor}));
  CheckTracingOutputs({new_rf_tensor}, trace);
  CheckTracingOutputs({new_rf_tensor}, ir_sch.GetTraceDesc());
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_MergeExprs) {
  auto funcs_0 = LowerCompute({32, 128}, target);
  auto funcs_1 =
      LowerCompute({32, 32, 32}, target, true, "elementwise-add_const");

  ir::IRSchedule ir_sch =
      ir::IRSchedule(ir::ModuleExpr({ir::ir_utils::IRCopy(funcs_0[0]->body),
                                     ir::ir_utils::IRCopy(funcs_0[0]->body)}));
  ir_sch.MergeExprs();
  trace.Append(ScheduleDesc::Step("MergeExprs", {}, {}, {}));
  ir::IRSchedule replay_sch =
      ir::IRSchedule(ir::ModuleExpr({ir::ir_utils::IRCopy(funcs_0[0]->body),
                                     ir::ir_utils::IRCopy(funcs_0[0]->body)}));
  trace.Replay(&replay_sch);

  auto lhs_exprs = ir_sch.GetModule().GetExprs();
  auto rhs_exprs = replay_sch.GetModule().GetExprs();
  ASSERT_EQ(lhs_exprs.size(), rhs_exprs.size());
  for (auto i = 0; i < lhs_exprs.size(); ++i) {
    ASSERT_EQ(utils::GetStreamCnt(lhs_exprs.at(i)),
              utils::GetStreamCnt(rhs_exprs.at(i)));
  }
}

TEST_F(TestScheduleDesc, StepKind_Annotate) {
  lowered_funcs = LowerCompute({32, 128}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.Annotate(block_b, "k1", 64);
  trace.Append(ScheduleDesc::Step("AnnotateIntAttr",
                                  {{"block", std::vector<Expr>({block_b})}},
                                  {{"key", std::string("k1")}, {"value", 64}},
                                  {}));

  block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.Annotate(block_b, "k2", true);
  trace.Append(ScheduleDesc::Step("AnnotateBoolAttr",
                                  {{"block", std::vector<Expr>({block_b})}},
                                  {{"key", std::string("k2")}, {"value", true}},
                                  {}));

  block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.Annotate(block_b, "k3", 2.0f);
  trace.Append(ScheduleDesc::Step("AnnotateFloatAttr",
                                  {{"block", std::vector<Expr>({block_b})}},
                                  {{"key", std::string("k3")}, {"value", 2.0f}},
                                  {}));

  block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.Annotate(block_b, "k4", std::string("v4"));
  trace.Append(ScheduleDesc::Step(
      "AnnotateStringAttr",
      {{"block", std::vector<Expr>({block_b})}},
      {{"key", std::string("k4")}, {"value", std::string("v4")}},
      {}));

  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_Unannotate) {
  lowered_funcs = LowerCompute({32, 128}, target);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);

  auto block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.Annotate(block_b, "k1", 64);
  trace.Append(ScheduleDesc::Step("AnnotateIntAttr",
                                  {{"block", std::vector<Expr>({block_b})}},
                                  {{"key", std::string("k1")}, {"value", 64}},
                                  {}));

  block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.Annotate(block_b, "k2", true);
  trace.Append(ScheduleDesc::Step("AnnotateBoolAttr",
                                  {{"block", std::vector<Expr>({block_b})}},
                                  {{"key", std::string("k2")}, {"value", true}},
                                  {}));

  block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.Unannotate(block_b, "k1");
  trace.Append(ScheduleDesc::Step("Unannotate",
                                  {{"block", std::vector<Expr>({block_b})}},
                                  {{"key", std::string("k1")}},
                                  {}));

  block_b = ir_sch.GetBlock("B");
  trace.Append(ScheduleDesc::Step(
      "GetBlock", {}, {{"block_name", std::string("B")}}, {block_b}));
  ir_sch.Unannotate(block_b, "k2");
  trace.Append(ScheduleDesc::Step("Unannotate",
                                  {{"block", std::vector<Expr>({block_b})}},
                                  {{"key", std::string("k2")}},
                                  {}));

  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_SamplePerfectTile) {
  Expr M(1024);
  Var n(1, "n");

  Placeholder<int> A("A", {M});
  auto B = Compute(
      {M}, [&](Expr i) { return A(i) + n; }, "B");
  lowered_funcs = cinn::lang::LowerVec("test_sample_perfect_tile",
                                       CreateStages({A, B}),
                                       {A, B},
                                       {},
                                       {},
                                       nullptr,
                                       target,
                                       true);

  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);
  auto loops = ir_sch.GetLoops("B");
  trace.Append(ScheduleDesc::Step(
      "GetLoopsWithName", {}, {{"block_name", std::string("B")}}, loops));
  auto result = ir_sch.SamplePerfectTile(loops[0], 2, 64);
  std::vector<int> decision;
  std::transform(
      result.begin(), result.end(), std::back_inserter(decision), [](Expr x) {
        return x.as_int32();
      });
  trace.Append(ScheduleDesc::Step(
      "SamplePerfectTile",
      {{"loop", std::vector<Expr>({loops[0]})}},
      {{"n", 2}, {"max_innermost_factor", 64}, {"decision", decision}},
      result));
  CheckTracingOutputs(result, trace);
  CheckTracingOutputs(result, ir_sch.GetTraceDesc());
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

TEST_F(TestScheduleDesc, StepKind_SampleCategorical) {
  lowered_funcs = LowerCompute({32, 32, 64}, target, true);
  ir::IRSchedule ir_sch = MakeIRSchedule(lowered_funcs);
  Expr ret = ir_sch.SampleCategorical({1, 2, 3}, {1.0, 2.0, 3.0});
  std::vector<int> decision = {ret.as_int32()};
  trace.Append(
      ScheduleDesc::Step("SampleCategorical",
                         {},
                         {{"candidates", std::vector<int>({1, 2, 3})},
                          {"probs", std::vector<float>({1.0, 2.0, 3.0})},
                          {"decision", decision}},
                         {ret}));
  CheckTracingOutputs({ret}, trace);
  CheckTracingOutputs({ret}, ir_sch.GetTraceDesc());
  CheckReplayResult(ir_sch, trace);
  CheckReplayResult(ir_sch, ir_sch.GetTraceDesc());
}

}  // namespace ir
}  // namespace cinn
