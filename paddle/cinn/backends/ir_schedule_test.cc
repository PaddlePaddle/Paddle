// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/schedule/ir_schedule.h"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <tuple>
#include <vector>

#include "paddle/cinn/backends/codegen_c.h"
#include "paddle/cinn/backends/codegen_c_x86.h"
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/schedule/ir_schedule_error.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/remove_schedule_block.h"
#include "paddle/cinn/optim/unroll_loops.h"
#include "paddle/cinn/optim/vectorize_loops.h"

namespace cinn {
namespace backends {

TEST(IrSchedule, split_and_fuse1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_split_and_fuse1", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto fused = ir_sch.Fuse("B", {0, 1});
  auto splited = ir_sch.Split(fused, {4, -1});

  auto loops = ir_sch.GetLoops("B");
  fused = ir_sch.Fuse(loops);
  splited = ir_sch.Split(fused, {256, -1});

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_split_and_fuse1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i_j_fused_i_j_fused_0_fused = 0; i_j_fused_i_j_fused_0_fused < 256; i_j_fused_i_j_fused_0_fused += 1) {
    for (int32_t i_j_fused_i_j_fused_0_fused_0 = 0; i_j_fused_i_j_fused_0_fused_0 < 4; i_j_fused_i_j_fused_0_fused_0 += 1) {
      B[(((i_j_fused_i_j_fused_0_fused / 8) * 32) + (((4 * i_j_fused_i_j_fused_0_fused) + i_j_fused_i_j_fused_0_fused_0) & 31))] = A[(((i_j_fused_i_j_fused_0_fused / 8) * 32) + (((4 * i_j_fused_i_j_fused_0_fused) + i_j_fused_i_j_fused_0_fused_0) & 31))];
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, split_and_fuse2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_split_and_fuse2", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");

  auto fused = ir_sch.Fuse(loops);
  auto splited = ir_sch.Split(fused, {-1, 20});
  VLOG(3) << "After split {-1, 20}, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "split_and_fuse2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_split_and_fuse2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i_j_fused = 0; i_j_fused < 52; i_j_fused += 1) {
    for (int32_t i_j_fused_0 = 0; i_j_fused_0 < 20; i_j_fused_0 += 1) {
      if ((((20 * i_j_fused) + i_j_fused_0) < 1024)) {
        B[((20 * i_j_fused) + i_j_fused_0)] = A[((20 * i_j_fused) + i_j_fused_0)];
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

void TestSplitThrow() {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_split_throw", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(
      mod_expr, -1, false, utils::ErrorMessageLevel::kGeneral);
  auto fused = ir_sch.Fuse("B", {0, 1});
  // statement that cause the exception
  auto splited = ir_sch.Split(fused, {-1, -1});

  auto loops = ir_sch.GetLoops("B");
  fused = ir_sch.Fuse(loops);
  splited = ir_sch.Split(fused, {256, -1});

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
}
TEST(IrSchedule, split_throw) {
  ASSERT_THROW(TestSplitThrow(), utils::enforce::EnforceNotMet);
}

TEST(IrSchedule, reorder1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_reorder1", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto splited = ir_sch.Split("B", 0, {-1, 4});
  splited = ir_sch.Split("B", 2, {-1, 2});

  auto loops = ir_sch.GetLoops("B");
  ir_sch.Reorder({loops[4], loops[0]});

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "reorder1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_reorder1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t k = 0; k < 32; k += 1) {
    for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
      for (int32_t j = 0; j < 16; j += 1) {
        for (int32_t j_0 = 0; j_0 < 2; j_0 += 1) {
          for (int32_t i = 0; i < 8; i += 1) {
            B[((4096 * i) + ((1024 * i_0) + ((64 * j) + ((32 * j_0) + k))))] = A[((4096 * i) + ((1024 * i_0) + ((64 * j) + ((32 * j_0) + k))))];
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, reorder2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_reorder2", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto splited = ir_sch.Split("B", 0, {-1, 4});
  splited = ir_sch.Split("B", 2, {-1, 2});

  ir_sch.Reorder("B", {4, 2, 3, 1, 0});

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "reorder2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_reorder2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t k = 0; k < 32; k += 1) {
    for (int32_t j = 0; j < 16; j += 1) {
      for (int32_t j_0 = 0; j_0 < 2; j_0 += 1) {
        for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
          for (int32_t i = 0; i < 8; i += 1) {
            B[((4096 * i) + ((1024 * i_0) + ((64 * j) + ((32 * j_0) + k))))] = A[((4096 * i) + ((1024 * i_0) + ((64 * j) + ((32 * j_0) + k))))];
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, reorder3) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_reorder3", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto all_blocks = ir_sch.GetAllBlocks();
  auto loops = ir_sch.GetLoops(all_blocks[0]);

  auto splited = ir_sch.Split(loops[0], {-1, 5});
  splited = ir_sch.Split("B", 2, {-1, 2});

  ir_sch.Reorder("B", {3, 1, 2, 0, 4});

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "reorder3 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_reorder3(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t j_0 = 0; j_0 < 2; j_0 += 1) {
    for (int32_t i_0 = 0; i_0 < 5; i_0 += 1) {
      for (int32_t j = 0; j < 16; j += 1) {
        for (int32_t i = 0; i < 7; i += 1) {
          if ((((5 * i) + i_0) < 32)) {
            for (int32_t k = 0; k < 32; k += 1) {
              B[((5120 * i) + ((1024 * i_0) + ((64 * j) + ((32 * j_0) + k))))] = A[((5120 * i) + ((1024 * i_0) + ((64 * j) + ((32 * j_0) + k))))];
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, reorder4) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_reorder4", stages, {A, B}, {}, {}, nullptr, target, true);
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto all_blocks = ir_sch.GetAllBlocks();
  auto block_b = ir_sch.GetBlock("B");
  auto loops = ir_sch.GetLoops(block_b);

  auto splited = ir_sch.Split("B", 0, {-1, 10});
  splited = ir_sch.Split("B", 2, {-1, 5});

  ir_sch.Reorder("B", {0, 2, 1, 3, 4});

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(3) << "reorder4 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_reorder4(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t j = 0; j < 7; j += 1) {
      for (int32_t i_0 = 0; i_0 < 10; i_0 += 1) {
        if ((((10 * i) + i_0) < 32)) {
          for (int32_t j_0 = 0; j_0 < 5; j_0 += 1) {
            if ((((5 * j) + j_0) < 32)) {
              for (int32_t k = 0; k < 32; k += 1) {
                B[((10240 * i) + ((1024 * i_0) + ((160 * j) + ((32 * j_0) + k))))] = A[((10240 * i) + ((1024 * i_0) + ((160 * j) + ((32 * j_0) + k))))];
              };
            };
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

#ifdef CINN_USE_OPENMP
TEST(IrSchedule, parallel) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  auto func = cinn::lang::LowerVec(
      "test_parallel", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK(!loops.empty());
  ir_sch.Parallel(loops[0]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_parallel(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  int num_task = max_concurrency();
  omp_set_num_threads(num_task);
  auto flambda = [=](int task_id, int num_task) -> int {
    int n_per_task = (((32 + num_task) - 1) / num_task);
    for (int32_t i = (task_id * n_per_task); i < 32 && i < ((task_id + 1) * n_per_task); i += 1) {
      for (int32_t j = 0; j < 32; j += 1) {
        B[((32 * i) + j)] = A[((32 * i) + j)];
      };
    }
    return 0;
  };
#pragma omp parallel num_threads(num_task)
  {
    int task_id = omp_get_thread_num();
    flambda(task_id, num_task);
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}
#endif  // CINN_USE_OPENMP

TEST(IrSchedule, vectorize) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  auto func = cinn::lang::LowerVec(
      "test_vectorize", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 2U);
  ir_sch.Vectorize(loops[1], 16);
  std::string origin = utils::GetStreamCnt(func[0]);
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_vectorize (_A, _B)
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      vectorize[16] for (j, 0, 32)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
}
)ROC"));
  optim::VectorizeLoops(&func[0]->body, target);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_vectorize(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 2; j += 1) {
      B[StackVec<16,int32_t>::Ramp(((32 * i) + (16 * j)), 1, 16)] = StackedVec<float,16>::Load(A,((32 * i) + (16 * j)));
    };
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, unroll) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  auto func = cinn::lang::LowerVec(
      "test_unroll", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 2U);
  ir_sch.Unroll(loops[1]);
  std::string origin = utils::GetStreamCnt(func[0]);
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_unroll (_A, _B)
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      unroll for (j, 0, 2)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
}
)ROC"));
  optim::UnrollLoop(&func[0]->body);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_unroll(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    B[(2 * i)] = A[(2 * i)];
    B[(1 + (2 * i))] = A[(1 + (2 * i))];
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, bind) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto stages = CreateStages({A, B});
  auto func = cinn::lang::LowerVec(
      "test_bind", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 2U);
  ir_sch.Bind(loops[0], "blockIdx.x");
  std::string origin = utils::GetStreamCnt(func[0]);
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_bind (_A, _B)
{
  ScheduleBlock(root)
  {
    thread_bind[blockIdx.x] for (i, 0, 32)
    {
      serial for (j, 0, 2)
      {
        ScheduleBlock(B)
        {
          i0, i1 = axis.bind(i, j)
          B[i0, i1] = A[i0, i1]
        }
      }
    }
  }
}
)ROC"));
}

TEST(IrSchedule, simple_compute_at) {
  Context::Global().ResetNameId();
  Expr M(128);
  Expr N(10);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_simple_compute_at", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto fused = ir_sch.Fuse("B", {0, 1});
  auto splited = ir_sch.Split(fused, {-1, 1024});

  fused = ir_sch.Fuse("C", {0, 1});
  splited = ir_sch.Split(fused, {-1, 1024});
  auto block_b = ir_sch.GetBlock("B");
  ir_sch.SimpleComputeAt(block_b, splited[1]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "simple_compute_at source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_simple_compute_at(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 128, 10 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i_j_fused_1 = 0; i_j_fused_1 < 2; i_j_fused_1 += 1) {
    for (int32_t i_j_fused_2 = 0; i_j_fused_2 < 1024; i_j_fused_2 += 1) {
      if ((((1024 * i_j_fused_1) + i_j_fused_2) < 1280)) {
        B[((1024 * i_j_fused_1) + i_j_fused_2)] = A[((1024 * i_j_fused_1) + i_j_fused_2)];
        C[((1024 * i_j_fused_1) + i_j_fused_2)] = B[((1024 * i_j_fused_1) + i_j_fused_2)];
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at0) {
  Context::Global().ResetNameId();
  Expr M(128);
  Expr N(10);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_compute_at0", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto fused = ir_sch.Fuse("B", {0, 1});
  auto splited = ir_sch.Split(fused, {-1, 1024});

  fused = ir_sch.Fuse("C", {0, 1});
  splited = ir_sch.Split(fused, {-1, 1024});
  auto block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeAt(block_b, splited[1]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at0 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at0(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 128, 10 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i_j_fused_1 = 0; i_j_fused_1 < 2; i_j_fused_1 += 1) {
    for (int32_t i_j_fused_2 = 0; i_j_fused_2 < 1024; i_j_fused_2 += 1) {
      if ((((1024 * i_j_fused_1) + i_j_fused_2) < 1280)) {
        B[((1024 * i_j_fused_1) + i_j_fused_2)] = A[((1024 * i_j_fused_1) + i_j_fused_2)];
        C[((1024 * i_j_fused_1) + i_j_fused_2)] = B[((1024 * i_j_fused_1) + i_j_fused_2)];
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(i, j, k); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_compute_at1", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[1]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t ax0 = 0; ax0 < 32; ax0 += 1) {
        B[((1024 * i) + ((32 * j) + ax0))] = A[((1024 * i) + ((32 * j) + ax0))];
      };
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = B[((1024 * i) + ((32 * j) + k))];
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at2) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {M, M}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {N, N}, [&](Var i, Var j) { return B(i + j, i + j); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_compute_at2", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[0]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 64, 64 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t ax0 = 0; ax0 < 32; ax0 += 1) {
      for (int32_t ax1 = 0; ax1 < 32; ax1 += 1) {
        B[((64 * ax0) + ((64 * i) + (ax1 + i)))] = A[((64 * ax0) + ((64 * i) + (ax1 + i)))];
      };
    };
    for (int32_t j = 0; j < 32; j += 1) {
      C[((32 * i) + j)] = B[((65 * i) + (65 * j))];
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at3) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {M, M}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {M, M}, [&](Var i, Var j) { return B(i, j); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_compute_at3", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");

  auto fused = ir_sch.Fuse("C", {0, 1});
  auto splited = ir_sch.Split(fused, {32, -1});

  auto loops = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[0]);

  VLOG(1) << "After ComputeAt, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at3 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_at3(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 64, 64 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i_j_fused = 0; i_j_fused < 32; i_j_fused += 1) {
    for (int32_t ax0 = 0; ax0 < 2; ax0 += 1) {
      for (int32_t ax1 = 0; ax1 < 64; ax1 += 1) {
        B[((64 * ax0) + ((128 * i_j_fused) + ax1))] = A[((64 * ax0) + ((128 * i_j_fused) + ax1))];
      };
    };
    for (int32_t i_j_fused_0 = 0; i_j_fused_0 < 128; i_j_fused_0 += 1) {
      C[((128 * i_j_fused) + i_j_fused_0)] = B[((128 * i_j_fused) + i_j_fused_0)];
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

#ifdef CINN_WITH_CUDA
TEST(IrSchedule, compute_at4) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");
  auto C = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return B(i, j, k); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec(
      "test_compute_at4", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[1]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at4 source code is :\n" << source_code;

  std::string target_code = codegen.GetSourceHeader() + R"ROC(__global__
void test_compute_at4(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 32768 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t ax0 = 0; ax0 < 32; ax0 += 1) {
        B[((1024 * i) + ((32 * j) + ax0))] = A[((1024 * i) + ((32 * j) + ax0))];
      };
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = B[((1024 * i) + ((32 * j) + k))];
      };
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at5) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {M, M}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {N, N}, [&](Var i, Var j) { return B(i + j, i + j); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec(
      "test_compute_at5", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[0]);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at5 source code is :\n" << source_code;

  std::string target_code = codegen.GetSourceHeader() +
                            R"ROC(__global__
void test_compute_at5(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 4096 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t ax0 = 0; ax0 < 32; ax0 += 1) {
      for (int32_t ax1 = 0; ax1 < 32; ax1 += 1) {
        B[((64 * ax0) + ((64 * i) + (ax1 + i)))] = A[((64 * ax0) + ((64 * i) + (ax1 + i)))];
      };
    };
    for (int32_t j = 0; j < 32; j += 1) {
      C[((32 * i) + j)] = B[((65 * i) + (65 * j))];
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_at6) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {M, M}, [&](Var i, Var j) { return A(i, j); }, "B");
  auto C = Compute(
      {M, M}, [&](Var i, Var j) { return B(i, j); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec(
      "test_compute_at6", stages, {A, C}, {}, {}, nullptr, target, true);
  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");

  auto fused = ir_sch.Fuse("C", {0, 1});
  auto splited = ir_sch.Split(fused, {32, -1});

  auto loops = ir_sch.GetLoops("C");

  ir_sch.ComputeAt(block_b, loops[1]);

  VLOG(1) << "After ComputeAt, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_at6 source code is :\n" << source_code;

  std::string target_code = codegen.GetSourceHeader() + R"ROC(__global__
void test_compute_at6(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 4096 ];
  float* B = _B_temp_buffer;
  for (int32_t i_j_fused = 0; i_j_fused < 32; i_j_fused += 1) {
    for (int32_t i_j_fused_0 = 0; i_j_fused_0 < 128; i_j_fused_0 += 1) {
      B[((128 * i_j_fused) + i_j_fused_0)] = A[((128 * i_j_fused) + i_j_fused_0)];
      C[((128 * i_j_fused) + i_j_fused_0)] = B[((128 * i_j_fused) + i_j_fused_0)];
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}
#endif

TEST(IrSchedule, cache_read1) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);
  Expr P(16);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {N, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {P, P}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_cache_read1", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto a_cache = ir_sch.CacheRead(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto b_cache = ir_sch.CacheRead(block_c, 0, "local");

  VLOG(1) << "After CacheRead, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_read1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_read1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t cache_ax0 = 0; cache_ax0 < 32; cache_ax0 += 1) {
    for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
      A_local_temp_buffer[((64 * cache_ax0) + cache_ax1)] = A[((64 * cache_ax0) + cache_ax1)];
    };
  };
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      B[((32 * i) + j)] = (2.00000000f * A_local_temp_buffer[((64 * i) + j)]);
    };
  };
  for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 16; cache_ax0_0 += 1) {
    for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 16; cache_ax1_0 += 1) {
      B_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)] = B[((32 * cache_ax0_0) + cache_ax1_0)];
    };
  };
  for (int32_t i = 0; i < 16; i += 1) {
    for (int32_t j = 0; j < 16; j += 1) {
      C[((16 * i) + j)] = (1.00000000f + B_local_temp_buffer[((32 * i) + j)]);
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, cache_read2) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_cache_read2", stages, {A, B}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");

  auto a_cache = ir_sch.CacheRead(block_b, 0, "local");

  auto loops = ir_sch.GetLoops("B");
  ir_sch.ComputeAt(a_cache, loops[1]);

  VLOG(1) << "After CacheRead and ComputeAt, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_read2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_read2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      A_local_temp_buffer[((32 * i) + j)] = A[((32 * i) + j)];
      B[((32 * i) + j)] = (2.00000000f * A_local_temp_buffer[((32 * i) + j)]);
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, cache_write1) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_cache_write1", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto c_cache = ir_sch.CacheWrite(block_c, 0, "local");

  VLOG(1) << "After CacheWrite, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_write1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_write1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 64, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      B_local_temp_buffer[((32 * i) + j)] = (2.00000000f * A[((32 * i) + j)]);
    };
  };
  for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
    for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
      B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
    };
  };
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      C_local_temp_buffer[((32 * i) + j)] = (1.00000000f + B[((32 * i) + j)]);
    };
  };
  for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 64; cache_ax0_0 += 1) {
    for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 32; cache_ax1_0 += 1) {
      C[((32 * cache_ax0_0) + cache_ax1_0)] = C_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)];
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, cache_write2) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_cache_write2", stages, {A, B}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto loops = ir_sch.GetLoops("B");
  ir_sch.ComputeAt(b_cache, loops[1]);

  VLOG(1) << "After CacheWrite and ComputeAt, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_write2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_write2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
    for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
      B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)] = (2.00000000f * A[((32 * cache_ax0) + cache_ax1)]);
      B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
    };
  };
  cinn_buffer_free((void*)(0), _B);
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

#ifdef CINN_WITH_CUDA
TEST(IrSchedule, cache_read3) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);
  Expr P(16);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, M});
  auto B = Compute(
      {N, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {P, P}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec(
      "test_cache_read3", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto a_cache = ir_sch.CacheRead(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto b_cache = ir_sch.CacheRead(block_c, 0, "local");
  auto loops_c = ir_sch.GetLoops("C");
  ir_sch.SyncThreads(loops_c[1], false);
  auto loops_b = ir_sch.GetLoops("B");
  ir_sch.SyncThreads(loops_b[1]);

  VLOG(1) << "After CacheRead, IR is : " << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_read3 source code is :\n" << source_code;

  std::string target_code = codegen.GetSourceHeader() + R"ROC(__global__
void test_cache_read3(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 1024 ];
  float* B = _B_temp_buffer;
  for (int32_t cache_ax0 = 0; cache_ax0 < 32; cache_ax0 += 1) {
    for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
      A_local_temp_buffer[((64 * cache_ax0) + cache_ax1)] = A[((64 * cache_ax0) + cache_ax1)];
    };
  };
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      B[((32 * i) + j)] = (2.00000000f * A_local_temp_buffer[((64 * i) + j)]);
    };
    __syncthreads();
  };
  for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 16; cache_ax0_0 += 1) {
    for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 16; cache_ax1_0 += 1) {
      B_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)] = B[((32 * cache_ax0_0) + cache_ax1_0)];
    };
  };
  for (int32_t i = 0; i < 16; i += 1) {
    __syncthreads();
    for (int32_t j = 0; j < 16; j += 1) {
      C[((16 * i) + j)] = (1.00000000f + B_local_temp_buffer[((32 * i) + j)]);
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, cache_write3) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("shared");

  auto func = cinn::lang::LowerVec(
      "test_cache_write3", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto c_cache = ir_sch.CacheWrite(block_c, 0, "local");
  auto loops_c = ir_sch.GetLoops("C");
  ir_sch.SyncThreads(loops_c[0], false);
  auto loops_b = ir_sch.GetLoops("B");
  ir_sch.SyncThreads(loops_b[0]);

  VLOG(1) << "After CacheWrite, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_write3 source code is :\n" << source_code;

  std::string target_code = codegen.GetSourceHeader() + R"ROC(__global__
void test_cache_write3(const float* __restrict__ A, float* __restrict__ C)
{
  __shared__ float _B_temp_buffer [ 2048 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      B_local_temp_buffer[((32 * i) + j)] = (2.00000000f * A[((32 * i) + j)]);
    };
  };
  for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
    for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
      B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
    };
  };
  __syncthreads();
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      C_local_temp_buffer[((32 * i) + j)] = (1.00000000f + B[((32 * i) + j)]);
    };
  };
  __syncthreads();
  for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 64; cache_ax0_0 += 1) {
    for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 32; cache_ax1_0 += 1) {
      C[((32 * cache_ax0_0) + cache_ax1_0)] = C_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)];
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, sync_threads) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j) * Expr(2.f); }, "B");
  auto C = Compute(
      {M, N}, [&](Var i, Var j) { return B(i, j) + Expr(1.f); }, "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("shared");

  auto func = cinn::lang::LowerVec(
      "test_sync_threads", stages, {A, C}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto block_c = ir_sch.GetBlock("C");
  auto c_cache = ir_sch.CacheWrite(block_c, 0, "local");
  block_c = ir_sch.GetBlock("C");
  ir_sch.SyncThreads(block_c, false);
  block_b = ir_sch.GetBlock("B");
  ir_sch.SyncThreads(block_b);

  VLOG(1) << "After CacheWrite and SyncThreads, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = codegen.GetSourceHeader() + R"ROC(__global__
void test_sync_threads(const float* __restrict__ A, float* __restrict__ C)
{
  __shared__ float _B_temp_buffer [ 2048 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      B_local_temp_buffer[((32 * i) + j)] = (2.00000000f * A[((32 * i) + j)]);
    };
  };
  for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
    for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
      B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
      __syncthreads();
    };
  };
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      C_local_temp_buffer[((32 * i) + j)] = (1.00000000f + B[((32 * i) + j)]);
    };
  };
  for (int32_t cache_ax0_0 = 0; cache_ax0_0 < 64; cache_ax0_0 += 1) {
    for (int32_t cache_ax1_0 = 0; cache_ax1_0 < 32; cache_ax1_0 += 1) {
      __syncthreads();
      C[((32 * cache_ax0_0) + cache_ax1_0)] = C_local_temp_buffer[((32 * cache_ax0_0) + cache_ax1_0)];
    };
  };
}

)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}
#endif

TEST(IrSchedule, cache_write4) {
  Context::Global().ResetNameId();
  Expr M(64);
  Expr N(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, N});
  Var k(32, "k0");
  auto B = Compute(
      {M, N},
      [&](Var i, Var j) { return lang::ReduceSum(A(i, j, k), {k}); },
      "B");

  auto stages = CreateStages({A, B});

  auto func = cinn::lang::LowerVec(
      "test_cache_write4", stages, {A, B}, {}, {}, nullptr, target, true);

  CHECK_EQ(func.size(), 1U);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  auto loops = ir_sch.GetLoops("B");

  VLOG(1) << "After CacheWrite, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "cache_write4 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_cache_write4(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* B__reduce_init = ((float*)(_B->memory));
  for (int32_t i = 0; i < 64; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      B__reduce_init[((32 * i) + j)] = 0.00000000f;
      for (int32_t k0 = 0; k0 < 32; k0 += 1) {
        B_local_temp_buffer[((32 * i) + j)] = (B_local_temp_buffer[((32 * i) + j)] + A[((1024 * i) + ((32 * j) + k0))]);
      };
    };
  };
  for (int32_t cache_ax0 = 0; cache_ax0 < 64; cache_ax0 += 1) {
    for (int32_t cache_ax1 = 0; cache_ax1 < 32; cache_ax1 += 1) {
      B[((32 * cache_ax0) + cache_ax1)] = B_local_temp_buffer[((32 * cache_ax0) + cache_ax1)];
    };
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, rfactor) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);
  Expr K(16);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, K});
  Var j(2, "j0");
  Var k(16, "k0");
  auto B = Compute(
      {M},
      [&](Var i) {
        return lang::ReduceSum(A(i, j, k), {j, k});
      },
      "B");

  auto stages = CreateStages({A, B});
  auto func = cinn::lang::LowerVec(
      "test_rfactor", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 3U);
  auto new_rf_tensor = ir_sch.Rfactor(loops[2], 0);
  auto* new_rf_tensor_ref = new_rf_tensor.As<ir::_Tensor_>();
  CHECK(new_rf_tensor_ref);
  CHECK(new_rf_tensor_ref->buffer.defined());
  func[0]->temp_bufs.push_back(new_rf_tensor_ref->buffer);
  func[0]->PrepareBufferCastExprs();
  std::string origin = utils::GetStreamCnt(func[0]);
  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_rfactor (_A, _B)
{
  ScheduleBlock(root)
  {
    {
      serial for (rf_k0, 0, 16)
      {
        serial for (i, 0, 32)
        {
          ScheduleBlock(rf_B__reduce_init)
          {
            i0, i1_0 = axis.bind(i, rf_k0)
            rf_B__reduce_init[i1_0, i0] = 0.00000000f
          }
          serial for (j0, 0, 2)
          {
            ScheduleBlock(rf_B)
            {
              i0_0, i1, i2 = axis.bind(i, j0, rf_k0)
              rf_B[i2, i0_0] = (rf_B[i2, i0_0] + A[i0_0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        ScheduleBlock(B__reduce_init)
        {
          i0 = axis.bind(i)
          B__reduce_init[i0] = 0.00000000f
        }
        serial for (k0, 0, 16)
        {
          ScheduleBlock(B)
          {
            i0_0, i2 = axis.bind(i, k0)
            B[i0_0] = (B[i0_0] + rf_B[i2, i0_0])
          }
        }
      }
    }
  }
}
)ROC"));
  // optimze pass: add temp buffers
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_rfactor(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* rf__B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 16, 32 });
  cinn_buffer_malloc((void*)(0), _B);
  cinn_buffer_malloc((void*)(0), rf__B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* B__reduce_init = ((float*)(_B->memory));
  float* rf_B = ((float*)(rf__B->memory));
  float* rf_B__reduce_init = ((float*)(rf__B->memory));
  for (int32_t rf_k0 = 0; rf_k0 < 16; rf_k0 += 1) {
    for (int32_t i = 0; i < 32; i += 1) {
      rf_B__reduce_init[((32 * rf_k0) + i)] = 0.00000000f;
      for (int32_t j0 = 0; j0 < 2; j0 += 1) {
        rf_B[((32 * rf_k0) + i)] = (rf_B[((32 * rf_k0) + i)] + A[((32 * i) + ((16 * j0) + rf_k0))]);
      };
    };
  };
  for (int32_t i = 0; i < 32; i += 1) {
    B__reduce_init[i] = 0.00000000f;
    for (int32_t k0 = 0; k0 < 16; k0 += 1) {
      B[i] = (B[i] + rf_B[((32 * k0) + i)]);
    };
  };
  cinn_buffer_free((void*)(0), rf__B);
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, rfactor1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);
  Expr K(16);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, K});
  Var j(2, "j0");
  Var k(16, "k0");
  auto B = Compute(
      {M},
      [&](Var i) {
        return lang::ReduceSum(A(i, j, k), {j, k});
      },
      "B");

  auto stages = CreateStages({A, B});
  auto func = cinn::lang::LowerVec(
      "test_rfactor", stages, {A, B}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("B");
  CHECK_EQ(loops.size(), 3U);
  auto new_rf_tensor = ir_sch.Rfactor(loops[1], 1);
  auto* new_rf_tensor_ref = new_rf_tensor.As<ir::_Tensor_>();
  CHECK(new_rf_tensor_ref);
  CHECK(new_rf_tensor_ref->buffer.defined());
  func[0]->temp_bufs.push_back(new_rf_tensor_ref->buffer);
  func[0]->PrepareBufferCastExprs();
  std::string origin = utils::GetStreamCnt(func[0]);

  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_rfactor (_A, _B)
{
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (rf_j0, 0, 2)
        {
          ScheduleBlock(rf_B__reduce_init)
          {
            i0, i1_0 = axis.bind(i, rf_j0)
            rf_B__reduce_init[i0, i1_0] = 0.00000000f
          }
          serial for (k0, 0, 16)
          {
            ScheduleBlock(rf_B)
            {
              i0_0, i1, i2 = axis.bind(i, rf_j0, k0)
              rf_B[i0_0, i1] = (rf_B[i0_0, i1] + A[i0_0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        ScheduleBlock(B__reduce_init)
        {
          i0 = axis.bind(i)
          B__reduce_init[i0] = 0.00000000f
        }
        serial for (j0, 0, 2)
        {
          ScheduleBlock(B)
          {
            i0_0, i1 = axis.bind(i, j0)
            B[i0_0] = (B[i0_0] + rf_B[i0_0, i1])
          }
        }
      }
    }
  }
}
)ROC"));
  // optimze pass: add temp buffers
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_rfactor(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* rf__B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 2 });
  cinn_buffer_malloc((void*)(0), _B);
  cinn_buffer_malloc((void*)(0), rf__B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* B__reduce_init = ((float*)(_B->memory));
  float* rf_B = ((float*)(rf__B->memory));
  float* rf_B__reduce_init = ((float*)(rf__B->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t rf_j0 = 0; rf_j0 < 2; rf_j0 += 1) {
      rf_B__reduce_init[((2 * i) + rf_j0)] = 0.00000000f;
      for (int32_t k0 = 0; k0 < 16; k0 += 1) {
        rf_B[((2 * i) + rf_j0)] = (rf_B[((2 * i) + rf_j0)] + A[((32 * i) + ((16 * rf_j0) + k0))]);
      };
    };
  };
  for (int32_t i = 0; i < 32; i += 1) {
    B__reduce_init[i] = 0.00000000f;
    for (int32_t j0 = 0; j0 < 2; j0 += 1) {
      B[i] = (B[i] + rf_B[((2 * i) + j0)]);
    };
  };
  cinn_buffer_free((void*)(0), rf__B);
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, rfactor2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(2);
  Expr K(16);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, K});
  Placeholder<float> B("B", {K, N});
  Var k(16, "k0");
  auto C = Compute(
      {M, N},
      [&](Var i, Var j) { return lang::ReduceSum(A(i, k) * B(k, j), {k}); },
      "C");

  auto stages = CreateStages({A, B, C});
  auto func = cinn::lang::LowerVec(
      "test_rfactor", stages, {A, B, C}, {}, {}, nullptr, target, true);
  CHECK(!func.empty());
  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  auto loops = ir_sch.GetLoops("C");
  CHECK_EQ(loops.size(), 3U);
  auto new_rf_tensor = ir_sch.Rfactor(loops[2], 0);
  auto* new_rf_tensor_ref = new_rf_tensor.As<ir::_Tensor_>();
  CHECK(new_rf_tensor_ref);
  CHECK(new_rf_tensor_ref->buffer.defined());
  func[0]->temp_bufs.push_back(new_rf_tensor_ref->buffer);
  func[0]->PrepareBufferCastExprs();
  std::string origin = utils::GetStreamCnt(func[0]);

  EXPECT_EQ(origin, utils::Trim(R"ROC(
function test_rfactor (_A, _B, _C)
{
  ScheduleBlock(root)
  {
    {
      serial for (rf_k0, 0, 16)
      {
        serial for (i, 0, 32)
        {
          serial for (j, 0, 2)
          {
            ScheduleBlock(rf_C__reduce_init)
            {
              i0, i1, i2_0 = axis.bind(i, j, rf_k0)
              rf_C__reduce_init[i2_0, i0, i1] = 0.00000000f
            }
            ScheduleBlock(rf_C)
            {
              i0_0, i1_0, i2 = axis.bind(i, j, rf_k0)
              rf_C[i2, i0_0, i1_0] = (rf_C[i2, i0_0, i1_0] + (A[i0_0, i2] * B[i2, i1_0]))
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 2)
        {
          ScheduleBlock(C__reduce_init)
          {
            i0, i1 = axis.bind(i, j)
            C__reduce_init[i0, i1] = 0.00000000f
          }
          serial for (k0, 0, 16)
          {
            ScheduleBlock(C)
            {
              i0_0, i1_0, i2 = axis.bind(i, j, k0)
              C[i0_0, i1_0] = (C[i0_0, i1_0] + rf_C[i2, i0_0, i1_0])
            }
          }
        }
      }
    }
  }
}
)ROC"));
  // optimze pass: add temp buffers
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_rfactor(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  const cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[2]));
  cinn_buffer_t* rf__C = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 16, 32, 2 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), rf__C);
  const float* A = ((const float*)(_A->memory));
  const float* B = ((const float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  float* C__reduce_init = ((float*)(_C->memory));
  float* rf_C = ((float*)(rf__C->memory));
  float* rf_C__reduce_init = ((float*)(rf__C->memory));
  for (int32_t rf_k0 = 0; rf_k0 < 16; rf_k0 += 1) {
    for (int32_t i = 0; i < 32; i += 1) {
      for (int32_t j = 0; j < 2; j += 1) {
        rf_C__reduce_init[((2 * i) + ((64 * rf_k0) + j))] = 0.00000000f;
        rf_C[((2 * i) + ((64 * rf_k0) + j))] = fma(A[((16 * i) + rf_k0)], B[((2 * rf_k0) + j)], rf_C[((2 * i) + ((64 * rf_k0) + j))]);
      };
    };
  };
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 2; j += 1) {
      C__reduce_init[((2 * i) + j)] = 0.00000000f;
      for (int32_t k0 = 0; k0 < 16; k0 += 1) {
        C[((2 * i) + j)] = (C[((2 * i) + j)] + rf_C[((2 * i) + ((64 * k0) + j))]);
      };
    };
  };
  cinn_buffer_free((void*)(0), rf__C);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_inline1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); },
      "B");
  auto C = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return B(j, i, k) * Expr(2.f); },
      "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_compute_inline1", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeInline(block_b);
  VLOG(1) << "After ComputeInline, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_inline1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_inline1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = fma(2.00000000f, A[((32 * i) + ((1024 * j) + k))], 2.00000000f);
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_inline2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); },
      "B");
  auto C = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return B(i, j, k) * Expr(2.f); },
      "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_compute_inline2", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops = ir_sch.GetLoops("C");
  ir_sch.ComputeAt(block_b, loops[1]);
  block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeInline(block_b);
  VLOG(1) << "After ComputeInline, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_inline2 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_inline2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = fma(2.00000000f, A[((1024 * i) + ((32 * j) + k))], 2.00000000f);
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

#ifdef CINN_WITH_CUDA
TEST(IrSchedule, compute_inline3) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); },
      "B");
  auto C = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return B(j, i, k) * Expr(2.f); },
      "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec(
      "test_compute_inline3", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeInline(block_b);
  VLOG(1) << "After ComputeInline, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);

  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_inline3 source code is :\n" << source_code;

  std::string target_code = codegen.GetSourceHeader() + R"ROC(__global__
void test_compute_inline3(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 32768 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = (2.00000000f + (2.00000000f * A[((32 * i) + ((1024 * j) + k))]));
      };
    };
  };
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, compute_inline4) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultNVGPUTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); },
      "B");
  auto C = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return B(i, j, k) * Expr(2.f); },
      "C");

  auto stages = CreateStages({A, B, C});
  stages[B]->SetBuffer("local");

  auto func = cinn::lang::LowerVec(
      "test_compute_inline4", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_b = ir_sch.GetBlock("B");
  auto loops = ir_sch.GetLoops("C");
  ir_sch.ComputeAt(block_b, loops[1]);
  block_b = ir_sch.GetBlock("B");
  ir_sch.ComputeInline(block_b);
  VLOG(1) << "After ComputeInline, IR is : "
          << ir_sch.GetModule().GetExprs().at(0);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenCUDA_Dev codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = codegen.GetSourceHeader() + R"ROC(__global__
void test_compute_inline4(const float* __restrict__ A, float* __restrict__ C)
{
  float _B_temp_buffer [ 32768 ];
  float* B = _B_temp_buffer;
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((1024 * i) + ((32 * j) + k))] = (2.00000000f + (2.00000000f * A[((1024 * i) + ((32 * j) + k))]));
      };
    };
  };
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}
#endif

TEST(IrSchedule, reverse_compute_inline1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(64);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return Expr(1.f) + A(i, j); }, "B");
  auto C = Compute(
      {N, M}, [&](Var i, Var j) { return Expr(2.f) * B(j, i); }, "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_compute_inline1", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_c = ir_sch.GetBlock("C");
  ir_sch.ReverseComputeInline(block_c);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_inline1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_inline1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 64 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 64; j += 1) {
      C[((32 * j) + i)] = fma(2.00000000f, A[((64 * i) + j)], 2.00000000f);
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, reverse_compute_inline2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return Expr(1.f) + A(i, j, k); },
      "B");
  auto C = Compute(
      {N, M, P},
      [&](Var i, Var j, Var k) { return Expr(2.f) * B(j, i, k); },
      "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_compute_inline1", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_c = ir_sch.GetBlock("C");
  ir_sch.ReverseComputeInline(block_c);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  VLOG(1) << "compute_inline1 source code is :\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_compute_inline1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 32; i += 1) {
    for (int32_t j = 0; j < 32; j += 1) {
      for (int32_t k = 0; k < 32; k += 1) {
        C[((32 * i) + ((1024 * j) + k))] = fma(2.00000000f, A[((1024 * i) + ((32 * j) + k))], 2.00000000f);
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, copytransform1) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); },
      "B");
  auto C = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return B(j, i, k) * Expr(2.f); },
      "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_copytransform1", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_c = ir_sch.GetBlock("C");
  auto loops_c = ir_sch.GetLoops(block_c);
  auto splited = ir_sch.Split(loops_c[1], {-1, 4});
  block_c = ir_sch.GetBlock("C");
  loops_c = ir_sch.GetLoops(block_c);
  splited = ir_sch.Split(loops_c[0], {-1, 8});

  auto block_b = ir_sch.GetBlock("B");
  block_c = ir_sch.GetBlock("C");

  ir_sch.CopyTransformAndLoopInfo(block_b, block_c);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_copytransform1(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 32, 32 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t i_0 = 0; i_0 < 8; i_0 += 1) {
      for (int32_t j = 0; j < 8; j += 1) {
        for (int32_t j_0 = 0; j_0 < 4; j_0 += 1) {
          for (int32_t k = 0; k < 32; k += 1) {
            B[((8192 * i) + ((1024 * i_0) + ((128 * j) + ((32 * j_0) + k))))] = (1.00000000f + A[((8192 * i) + ((1024 * i_0) + ((128 * j) + ((32 * j_0) + k))))]);
          };
        };
      };
    };
  };
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t i_0 = 0; i_0 < 8; i_0 += 1) {
      for (int32_t j = 0; j < 8; j += 1) {
        for (int32_t j_0 = 0; j_0 < 4; j_0 += 1) {
          for (int32_t k = 0; k < 32; k += 1) {
            C[((8192 * i) + ((1024 * i_0) + ((128 * j) + ((32 * j_0) + k))))] = (2.00000000f * B[((256 * i) + ((32 * i_0) + ((4096 * j) + ((1024 * j_0) + k))))]);
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, copytransform2) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(64);
  Expr P(128);

  Target target = common::DefaultHostTarget();

  Placeholder<float> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P},
      [&](Var i, Var j, Var k) { return A(i, j, k) + Expr(1.f); },
      "B");
  auto C = Compute(
      {M, M, P},
      [&](Var i, Var j, Var k) { return B(i, j, k) * Expr(2.f); },
      "C");

  auto stages = CreateStages({A, B, C});

  auto func = cinn::lang::LowerVec(
      "test_copytransform2", stages, {A, C}, {}, {}, nullptr, target, true);

  auto ast_expr = func[0]->body;
  std::vector<Expr> vec_ast{ast_expr};
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);

  auto block_c = ir_sch.GetBlock("C");
  auto loops_c = ir_sch.GetLoops(block_c);
  auto splited = ir_sch.Split(loops_c[1], {-1, 4});
  block_c = ir_sch.GetBlock("C");
  loops_c = ir_sch.GetLoops(block_c);
  splited = ir_sch.Split(loops_c[0], {-1, 8});

  auto block_b = ir_sch.GetBlock("B");
  block_c = ir_sch.GetBlock("C");
  ir_sch.CopyTransformAndLoopInfo(block_b, block_c);
  Module::Builder builder("module1", target);
  for (auto& i : func) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void test_copytransform2(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _C = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_t* _B = cinn_buffer_t::new_((cinn_device_kind_t)(0)/*target*/, cinn_float32_t(), { 32, 64, 128 });
  cinn_buffer_malloc((void*)(0), _C);
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* C = ((float*)(_C->memory));
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t i_0 = 0; i_0 < 8; i_0 += 1) {
      for (int32_t j = 0; j < 64; j += 1) {
        for (int32_t k = 0; k < 128; k += 1) {
          B[((65536 * i) + ((8192 * i_0) + ((128 * j) + k)))] = (1.00000000f + A[((65536 * i) + ((8192 * i_0) + ((128 * j) + k)))]);
        };
      };
    };
  };
  for (int32_t i = 0; i < 4; i += 1) {
    for (int32_t i_0 = 0; i_0 < 8; i_0 += 1) {
      for (int32_t j = 0; j < 8; j += 1) {
        for (int32_t j_0 = 0; j_0 < 4; j_0 += 1) {
          for (int32_t k = 0; k < 128; k += 1) {
            C[((32768 * i) + ((4096 * i_0) + ((512 * j) + ((128 * j_0) + k))))] = (2.00000000f * B[((65536 * i) + ((8192 * i_0) + ((512 * j) + ((128 * j_0) + k))))]);
          };
        };
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
  cinn_buffer_free((void*)(0), _C);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, Annotate) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto funcs = cinn::lang::LowerVec("test_annotate",
                                    CreateStages({A, B}),
                                    {A, B},
                                    {},
                                    {},
                                    nullptr,
                                    common::DefaultHostTarget(),
                                    true);
  ir::IRSchedule ir_sch(ir::ModuleExpr({funcs[0]->body}));
  auto fused = ir_sch.Fuse("B", {0, 1});
  auto block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k1", 64);
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k2", true);
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k3", 2.0f);
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k4", std::string("v4"));
  std::string expected_expr = R"ROC({
  ScheduleBlock(root)
  {
    serial for (i_j_fused, 0, 1024)
    {
      ScheduleBlock(B)
      {
        i0, i1 = axis.bind((i_j_fused / 32), (i_j_fused % 32))
        attrs(k1:64, k2:1, k3:2, k4:v4)
        B[i0, i1] = A[i0, i1]
      }
    }
  }
})ROC";
  ASSERT_EQ(utils::GetStreamCnt(ir_sch.GetModule().GetExprs().front()),
            expected_expr);
}

TEST(IrSchedule, Unannotate) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Placeholder<float> A("A", {M, N});
  auto B = Compute(
      {M, N}, [&](Var i, Var j) { return A(i, j); }, "B");

  auto funcs = cinn::lang::LowerVec("test_unannotate",
                                    CreateStages({A, B}),
                                    {A, B},
                                    {},
                                    {},
                                    nullptr,
                                    common::DefaultHostTarget(),
                                    true);
  ir::IRSchedule ir_sch(ir::ModuleExpr({funcs[0]->body}));
  auto fused = ir_sch.Fuse("B", {0, 1});
  auto block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k1", 64);
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k2", true);
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k3", 2.0f);
  block_b = ir_sch.GetBlock("B");
  ir_sch.Annotate(block_b, "k4", std::string("v4"));
  block_b = ir_sch.GetBlock("B");
  ir_sch.Unannotate(block_b, "k1");
  block_b = ir_sch.GetBlock("B");
  ir_sch.Unannotate(block_b, "k2");
  block_b = ir_sch.GetBlock("B");
  ir_sch.Unannotate(block_b, "k3");
  block_b = ir_sch.GetBlock("B");
  ir_sch.Unannotate(block_b, "k4");
  std::string expected_expr = R"ROC({
  ScheduleBlock(root)
  {
    serial for (i_j_fused, 0, 1024)
    {
      ScheduleBlock(B)
      {
        i0, i1 = axis.bind((i_j_fused / 32), (i_j_fused % 32))
        B[i0, i1] = A[i0, i1]
      }
    }
  }
})ROC";
  ASSERT_EQ(utils::GetStreamCnt(ir_sch.GetModule().GetExprs().front()),
            expected_expr);
}

TEST(IrSchedule, ComplexIndices) {
  Target target = common::DefaultHostTarget();
  ir::Expr M(32);
  ir::Expr K(64);

  Placeholder<float> A("A", {M, K});
  Var k(K.as_int32(), "reduce_axis_k");
  ir::Tensor B = Compute(
      {M}, [&](Var i) { return ReduceSum(A(i, k), {k}); }, "B");

  poly::StageMap stages = CreateStages({B});
  std::vector<ir::LoweredFunc> funcs =
      lang::LowerVec("TestIrSchedule_ReduceSum",
                     stages,
                     {A, B},
                     {},
                     {},
                     nullptr,
                     target,
                     true);
  ir::IRSchedule ir_sch(ir::ModuleExpr({funcs[0]->body}));
  VLOG(3) << "Lowered Expr:" << ir_sch.GetModule().GetExprs().front();

  auto loops_b = ir_sch.GetLoops("B");
  CHECK_EQ(loops_b.size(), 2);
  ir_sch.Split("B", 0, {8, -1});
  ir_sch.Split(
      "B", 2, {32, -1});  // after first splited, loops size has added to 3
  VLOG(3) << "Splited Expr:" << ir_sch.GetModule().GetExprs().front();

  CHECK_EQ(ir_sch.GetLoops("B").size(), 4);
  ir_sch.Reorder("B", {2, 0, 3, 1});
  VLOG(3) << "Reordered Expr:\n" << ir_sch.GetModule().GetExprs().front();

  auto block_b = ir_sch.GetBlock("B");
  auto a_cache = ir_sch.CacheRead(
      block_b,
      1,
      "shared");  // actually the read_buffer A should be indexed by 0
  VLOG(3) << "CacheRead-A Expr:\n" << ir_sch.GetModule().GetExprs().front();

  loops_b = ir_sch.GetLoops("B");
  ir_sch.ComputeAt(a_cache, loops_b[0]);
  VLOG(3) << "A_cache-ComputeAt-B Expr:\n"
          << ir_sch.GetModule().GetExprs().front();

  block_b = ir_sch.GetBlock("B");
  auto b_cache = ir_sch.CacheWrite(block_b, 0, "local");
  VLOG(3) << "CacheWrite-B Expr:\n" << ir_sch.GetModule().GetExprs().front();

  auto loops_b_cache =
      ir_sch.GetLoops(b_cache.As<ir::ScheduleBlockRealize>()
                          ->schedule_block.As<ir::ScheduleBlock>()
                          ->name);
  block_b = ir_sch.GetBlock("B");
  ir_sch.ReverseComputeAt(block_b, loops_b_cache[1]);
  VLOG(3) << "B-ReverseComputeAt-B_cache Expr:\n"
          << ir_sch.GetModule().GetExprs().front();

  Module::Builder builder("module1", target);
  for (auto& i : funcs) {
    builder.AddFunction(i);
  }
  auto module = builder.Build();
  CodeGenC codegen(target);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(module, CodeGenC::OutputKind::CImpl);
  VLOG(3) << "scheduled source code:\n" << source_code;

  std::string target_code = R"ROC(
#include <cinn_runtime.h>
#include <stdio.h>

void TestIrSchedule_ReduceSum(void* _args, int32_t num_args)
{
  const cinn_buffer_t* _A = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[0]));
  cinn_buffer_t* _B = cinn_pod_value_to_buffer_p(&(((cinn_pod_value_t*)(_args))[1]));
  cinn_buffer_malloc((void*)(0), _B);
  const float* A = ((const float*)(_A->memory));
  float* B = ((float*)(_B->memory));
  float* B__reduce_init = ((float*)(_B->memory));
  for (int32_t i = 0; i < 8; i += 1) {
    for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
      B__reduce_init[((4 * i) + i_0)] = 0.00000000f;
    };
  };
  for (int32_t reduce_axis_k = 0; reduce_axis_k < 32; reduce_axis_k += 1) {
    for (int32_t ax0 = 0; ax0 < 32; ax0 += 1) {
      for (int32_t ax1 = 0; ax1 < 2; ax1 += 1) {
        A_shared_temp_buffer[((64 * ax0) + ((2 * reduce_axis_k) + ax1))] = A[((64 * ax0) + ((2 * reduce_axis_k) + ax1))];
      };
    };
    for (int32_t i = 0; i < 8; i += 1) {
      for (int32_t reduce_axis_k_0 = 0; reduce_axis_k_0 < 2; reduce_axis_k_0 += 1) {
        for (int32_t i_0 = 0; i_0 < 4; i_0 += 1) {
          B_local_temp_buffer[((4 * i) + i_0)] = (B_local_temp_buffer[((4 * i) + i_0)] + A_shared_temp_buffer[((256 * i) + ((64 * i_0) + ((2 * reduce_axis_k) + reduce_axis_k_0)))]);
        };
      };
      for (int32_t ax0_0 = 0; ax0_0 < 4; ax0_0 += 1) {
        B[((4 * i) + ax0_0)] = B_local_temp_buffer[((4 * i) + ax0_0)];
      };
    };
  };
  cinn_buffer_free((void*)(0), _B);
}
)ROC";
  ASSERT_EQ(utils::Trim(target_code), utils::Trim(source_code));
}

TEST(IrSchedule, SamplePerfectTile) {
  Context::Global().ResetNameId();
  Expr M(1024);
  Placeholder<int> A("A", {M});
  auto B = Compute(
      {M}, [&](Expr i) { return A(i) + 1; }, "B");
  poly::StageMap stages = CreateStages({A, B});

  auto funcs = cinn::lang::LowerVec("test_sampleperfecttile",
                                    stages,
                                    {A, B},
                                    {},
                                    {},
                                    nullptr,
                                    common::DefaultHostTarget(),
                                    true);

  ir::IRSchedule ir_sch(ir::ModuleExpr({funcs[0]->body}));
  auto loops_b = ir_sch.GetLoops("B");
  std::vector<Expr> result = ir_sch.SamplePerfectTile(loops_b[0], 3, 64);
  ASSERT_EQ(result.size(), 3);
}

TEST(IrSchedule, GetChildBlocks) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr K(32);
  Placeholder<float> A("A", {M, N, K});
  auto B = Compute(
      {M, N, K}, [&A](Var i, Var j, Var k) { return A(i, j, k); }, "B");
  auto C = Compute(
      {M, N, K}, [&B](Var i, Var j, Var k) { return B(i, j, k); }, "C");
  auto funcs = cinn::lang::LowerVec("test_getchildblocks",
                                    CreateStages({A, B, C}),
                                    {A, C},
                                    {},
                                    {},
                                    nullptr,
                                    common::DefaultHostTarget(),
                                    true);
  ir::IRSchedule ir_sch(ir::ModuleExpr({funcs[0]->body}));

  auto block_b = ir_sch.GetBlock("B");
  auto loops = ir_sch.GetLoops("C");
  ir_sch.ComputeAt(block_b, loops[1]);
  loops = ir_sch.GetLoops("B");
  auto root_block = ir_sch.GetRootBlock(loops[1]);

  std::string expected_expr = R"ROC(ScheduleBlock(B)
{
  i0, i1, i2 = axis.bind(i, j, (0 + ax0))
  attrs(compute_at_extra_var:ax0)
  B[i0, i1, i2] = A[i0, i1, i2]
}, ScheduleBlock(C)
{
  i0_0, i1_0, i2_0 = axis.bind(i, j, k)
  C[i0_0, i1_0, i2_0] = B[i0_0, i1_0, i2_0]
})ROC";
  ASSERT_EQ(utils::GetStreamCnt(ir_sch.GetChildBlocks(root_block)),
            expected_expr);
}

TEST(IrSchedule, SampleCategorical) {
  Context::Global().ResetNameId();
  Expr M(32);
  Expr N(32);
  Expr P(32);
  Placeholder<int> A("A", {M, N, P});
  auto B = Compute(
      {M, N, P}, [&](Var i, Var j, Var k) { return A(i, j, k); }, "B");
  poly::StageMap stages = CreateStages({A, B});
  std::vector<int> decision;
  auto funcs = cinn::lang::LowerVec("test_samplecategorical",
                                    stages,
                                    {A, B},
                                    {},
                                    {},
                                    nullptr,
                                    common::DefaultHostTarget(),
                                    true);

  ir::IRSchedule ir_sch(ir::ModuleExpr({funcs[0]->body}));
  Expr result =
      ir_sch.SampleCategorical({1, 2, 3}, {1.0, 2.0, 3.0}, {decision});
  ASSERT_EQ(result.type(), Int(32));
}

}  // namespace backends
}  // namespace cinn
