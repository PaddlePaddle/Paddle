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

#include "paddle/cinn/poly/stage.h"

#include <gtest/gtest.h>

#include <set>

#include "paddle/cinn/backends/llvm/codegen_llvm.h"
#include "paddle/cinn/backends/llvm/simple_jit.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/common/test_helper.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"

namespace cinn {
namespace poly {

// Create a call.
Expr CreateCall(const std::string& name, const std::vector<Expr>& args) {
  auto expr = ir::Call::Make(
      Float(32), name, args, {}, ir::CallType::CINN, ir::FunctionRef(), 0);
  return expr;
}

TEST(Stage, split) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j]: 0<=i,j<=100 }");

  auto ele = Stage::New(domain);
  auto _outer_inner_ = ele->Split(Iterator("i"), 4);  // NOLINT
  auto& outer = std::get<0>(_outer_inner_);
  auto& inner = std::get<1>(_outer_inner_);
  LOG(INFO) << ele->transform();
  EXPECT_EQ(utils::GetStreamCnt(ele->transform()),
            "{ S[i, j] -> S[i_outer, i_inner, j' = j] : (-i + i_inner) mod 4 = "
            "0 and -3 + i <= 4i_outer <= i and 0 <= "
            "i_inner <= 3 }");

  EXPECT_EQ(outer.id, "i_outer");
  EXPECT_EQ(inner.id, "i_inner");
}

TEST(Stage, tile) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);

  auto _outer0_inner0_outer1_inner1_ =
      ele->Tile(Iterator("i"), Iterator("j"), 4, 6);  // NOLINT
  auto& outer0 = std::get<0>(_outer0_inner0_outer1_inner1_);
  auto& inner0 = std::get<1>(_outer0_inner0_outer1_inner1_);
  auto& outer1 = std::get<2>(_outer0_inner0_outer1_inner1_);
  auto& inner1 = std::get<3>(_outer0_inner0_outer1_inner1_);
  LOG(INFO) << ele->transform();
  EXPECT_EQ(outer0.id, "i_outer");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(inner0.id, "i_inner");
  EXPECT_EQ(outer1.id, "j_outer");
  EXPECT_EQ(utils::GetStreamCnt(ele->transform()),
            "{ S[i, j, k] -> S[i_outer, i_inner, j_outer, j_inner, k' = k] : "
            "(-i + i_inner) mod 4 = 0 and (-j + j_inner) mod "
            "6 = 0 and -3 + i <= 4i_outer <= i and 0 <= i_inner <= 3 and -5 + "
            "j <= 6j_outer <= j and 0 <= j_inner <= 5 }");
}

TEST(Stage, reorder) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);
  Iterator i("i"), j("j"), k("k");
  ele->Reorder(std::vector<Iterator>{{i, k, j}});
  LOG(INFO) << ele->transform();
}

TEST(Stage, split_reorder) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);
  auto _outer_inner_ = ele->Split(Iterator("i"), 4);  // NOLINT
  auto& outer = std::get<0>(_outer_inner_);
  auto& inner = std::get<1>(_outer_inner_);

  Iterator i("i"), j("j"), k("k");
  ele->Reorder(std::vector<Iterator>{{outer, k, inner, j}});
  LOG(INFO) << ele->transform();
}

TEST(ComputeAtRelation, basic) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain0(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  isl::set domain1(ctx, "{ D[a,b,c,d]: 0<=a,b,c,d<=100 }");

  auto stage0 = Stage::New(domain0);
  auto stage1 = Stage::New(domain0);

  ComputeAtRelation relation;
  relation.stage = stage1;
  relation.level = 2;
  ASSERT_TRUE(relation.IsCompatible(stage0.get()));
}

TEST(Stage, Fuse) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);
  auto _outer_inner_ = ele->Split(Iterator("i"), 4);  // NOLINT
  auto& outer = std::get<0>(_outer_inner_);
  auto& inner = std::get<1>(_outer_inner_);
  LOG(INFO) << "split: " << ele->transform();
  ele->Fuse(outer, inner);
  LOG(INFO) << "fused: " << ele->transform();
}

TEST(Stage, Fuse1) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx, "{ S[i,j,k]: 0<=i,j,k<=100 }");
  auto ele = Stage::New(domain);
  Iterator i("i");
  Iterator j("j");
  auto n = ele->Fuse(i, j);
  LOG(INFO) << "fused: " << ele->transform();
}

TEST(ComputeAt, Before) {
  Expr M(100), N(200);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto A_cache = Compute(
      {M, N}, [&](Expr i, Expr j) { return A(i, j); }, "cache");
  auto C = Compute(
      {Expr(10), Expr(10)},
      [&](Expr i, Expr j) { return A_cache(i, j) + B(i, j); },
      "C");

  auto stages = CreateStages({A_cache, C});

  stages[A_cache]->ComputeAt2(stages[C], 1);

  auto fn = Lower("fn", stages, {A, B, A_cache, C});
  LOG(INFO) << "fn:\n" << fn;

  auto target = R"ROC(
function fn (_A, _B, _cache, _C)
{
  serial for (i, 0, 10)
  {
    serial for (j, 0, 10)
    {
      cache[i, j] = A[i, j]
      C[i, j] = (cache[i, j] + B[i, j])
    }
  }
}
)ROC";

  ASSERT_EQ(utils::Trim(target), utils::GetStreamCnt(fn));
}

TEST(ComputeAt, simple) {
  {
    Expr n(64);
    auto A = Placeholder<float>("A", {n, n});

    auto A1 = Compute(
        {n, n}, [&](Expr i, Expr j) { return A(i, j); }, "A1");
    auto B = Compute(
        {n / 2, n / 2},
        [&](Expr i, Expr j) { return A1(i, j) + A1(i + 1, j) + A1(i + 2, j); },
        "B");

    auto stages = CreateStages({B});
    stages[B]->Split(0, 16);
    stages[A1]->ComputeAt2(stages[B], 1);

    auto fn = Lower("fn", stages, {A, A1, B});
    LOG(INFO) << "fn:\n" << fn;

    auto target = R"ROC(
function fn (_A, _A1, _B)
{
  serial for (i_outer, 0, 2)
  {
    serial for (i_inner, 0, 16)
    {
      serial for (j, 0, 64)
      {
        serial for (i_at, 0, 3)
        {
          A1[((16 * i_outer) + (i_at + i_inner)), j] = A[((16 * i_outer) + (i_at + i_inner)), j]
        }
      }
      serial for (j, 0, 32)
      {
        B[((16 * i_outer) + i_inner), j] = (A1[((16 * i_outer) + i_inner), j] + (A1[(1 + ((16 * i_outer) + i_inner)), j] + A1[(2 + ((16 * i_outer) + i_inner)), j]))
      }
    }
  }
}
)ROC";
    ASSERT_EQ(utils::Trim(target), utils::GetStreamCnt(fn));

    Module::Builder builder("module", common::DefaultHostTarget());
    builder.AddFunction(fn);

    CodeGenC codegen(common::DefaultHostTarget());
    codegen.SetInlineBuiltinCodes(false);
    LOG(INFO) << "source:\n"
              << codegen.Compile(builder.Build(),
                                 backends::CodeGenC::OutputKind::CImpl);
  }
}

TEST(ComputeAt, Before1) {
  Expr M(100), N(200);

  Placeholder<float> A("A", {M, N});

  auto create_module = [&] {
    // cached compute way
    auto cache_prepare = Compute({M, N} /*domain*/,
                                 [&](Var i, Var j) { return A(i, j); },
                                 "cache",
                                 {N} /*shape*/);

    auto transformed_compute = Compute(
        {M, N}, [&](Var i, Var j) { return Expr(1.f); }, "transformed");

    return std::make_tuple(cache_prepare, transformed_compute);
  };

  {  // C_init Before C
    auto _cache_prepare_transformed_compute_ = create_module();
    auto& cache_prepare = std::get<0>(_cache_prepare_transformed_compute_);
    auto& transformed_compute =
        std::get<1>(_cache_prepare_transformed_compute_);

    auto stages = CreateStages({cache_prepare, transformed_compute});
    stages[cache_prepare]->ComputeAt2(stages[transformed_compute], 1);

    // codegen and compare
    auto fn = Lower("fn", stages, {A, cache_prepare, transformed_compute});
    LOG(INFO) << "fn1 :\n" << fn;

    auto target = utils::Trim(R"ROC(
function fn (_A, _cache, _transformed)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 200)
    {
      cache[i] = A[i, j]
      transformed[i, j] = 1.00000000f
    }
  }
}
)ROC");

    ASSERT_EQ(utils::Trim(utils::GetStreamCnt(fn)), target);
  }
  {  // C_init After C
    auto _cache_prepare_transformed_compute_ = create_module();
    auto& cache_prepare = std::get<0>(_cache_prepare_transformed_compute_);
    auto& transformed_compute =
        std::get<1>(_cache_prepare_transformed_compute_);

    auto stages = CreateStages({cache_prepare, transformed_compute});
    stages[transformed_compute]->ComputeAt2(stages[cache_prepare], 1);

    // codegen and compare
    auto fn = Lower("fn", stages, {A, cache_prepare, transformed_compute});
    LOG(INFO) << "fn2 :\n" << fn;
    auto target = utils::Trim(R"ROC(
function fn (_A, _cache, _transformed)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 200)
    {
      transformed[i, j] = 1.00000000f
      cache[i] = A[i, j]
    }
  }
}
)ROC");

    ASSERT_EQ(utils::Trim(utils::GetStreamCnt(fn)), target);
  }
}

void TestElementwiseAddJitPrecession(
    std::function<void(ir::Tensor*, StageMap)>&& scheduler) {
  Expr M(30);
  Expr N(40);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  auto C = Compute(
      {M, N}, [&](Var i, Var j) -> Expr { return A(i, j) + B(i, j); }, "C");

  auto stages = CreateStages({C});

  scheduler(&C, stages);

  auto fn = Lower("fn", stages, {A, B, C});
  LOG(INFO) << "fn:\n" << fn;

  Module::Builder module_builder("some_module", common::DefaultHostTarget());
  module_builder.AddFunction(fn);

  auto jit = backends::SimpleJIT::Create();
  jit->Link(module_builder.Build(), false);
  auto _fn_handler = jit->Lookup("fn");
  auto* fn_handler = reinterpret_cast<lower_func_ptr_t>(_fn_handler);

  // create buffer and args
  auto A_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                   .set_random()
                   .Build();
  auto B_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                   .set_random()
                   .Build();
  auto C_buf = common::BufferBuilder(Float(32), {M.as_int32(), N.as_int32()})
                   .set_zero()
                   .Build();
  auto arg_pack =
      common::ArgsBuilder().Add(A_buf).Add(B_buf).Add(C_buf).Build();

  fn_handler(arg_pack.data(), arg_pack.size());

  auto* A_data = reinterpret_cast<float*>(A_buf->memory);
  auto* B_data = reinterpret_cast<float*>(B_buf->memory);
  auto* C_data = reinterpret_cast<float*>(C_buf->memory);
  for (int i = 0; i < A_buf->num_elements(); i++) {
    if (i < 4) LOG(INFO) << C_data[i];
    ASSERT_NEAR(A_data[i] + B_data[i], C_data[i], 1e-5);
  }

  cinn_buffer_free(nullptr, A_buf);
  cinn_buffer_free(nullptr, B_buf);
}

// use an elementwise_add to test fuse precision
TEST(Fuse, jit_precision_test) {
  TestElementwiseAddJitPrecession(
      [](ir::Tensor* C, StageMap stages) { stages[(*C)]->Fuse(0, 1); });
}

// split test fuse precision
TEST(Fuse, jit_precision_test2) {
  TestElementwiseAddJitPrecession([](ir::Tensor* C, StageMap stages) {
    auto _i_outer_i_inner_ = stages[(*C)]->Split(0, 4);
    auto& i_outer = std::get<0>(_i_outer_i_inner_);
    auto& i_inner = std::get<1>(_i_outer_i_inner_);
    stages[(*C)]->Fuse(i_outer, i_inner);
  });
}

TEST(Tile, jit_precision_test) {
  TestElementwiseAddJitPrecession(
      [](ir::Tensor* C, StageMap stages) { stages[(*C)]->Tile(0, 1, 4, 4); });
}

TEST(Reorder, jit_precision_test) {
  TestElementwiseAddJitPrecession([](ir::Tensor* C, StageMap stages) {
    auto* stage = stages[(*C)];
    stage->Reorder({stage->axis(1), stage->axis(0)});
  });
}

TEST(Unroll, jit_precision_test) {
  TestElementwiseAddJitPrecession(
      [](ir::Tensor* C, StageMap stages) { stages[(*C)]->Unroll(1); });
}

TEST(Unroll, jit_precision_test1) {
  TestElementwiseAddJitPrecession(
      [](ir::Tensor* C, StageMap stages) { stages[*C]->Unroll(0); });
}

TEST(ComputeInline, basic) {
  Expr M(100), N(200);
  Placeholder<float> A("A", {M, N});

  auto B = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return A(i, j) + 1.f; }, "B");
  auto B1 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B(i, j) + 1.f; }, "B1");
  auto B2 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B1(i, j) + 1.f; }, "B2");

  auto C = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B2(i, j) * 2.f; }, "C");

  auto stages = CreateStages({C});

  stages[B]->ComputeInline();
  stages[B1]->ComputeInline();
  stages[B2]->ComputeInline();

  auto inlined_B = B->inline_expanded({Expr(2), Expr(1)});
  ASSERT_EQ("(A[2, 1] + 1.00000000f)", utils::GetStreamCnt(inlined_B));

  auto fn = Lower("fn", stages, {A, C});

  LOG(INFO) << "fn:\n" << fn;

  auto target = R"ROC(
function fn (_A, _C)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 200)
    {
      C[i, j] = (6.00000000f + (2.00000000f * A[i, j]))
    }
  }
}
  )ROC";

  ASSERT_EQ(utils::Trim(target), utils::Trim(utils::GetStreamCnt(fn)));
}

TEST(ComputeInline, complex_graph) {
  Expr M(100), N(200);
  Placeholder<float> A("A", {M, N});

  auto B = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return A(i, j) + 1.f; }, "B");
  auto B1 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B(i, j) + 1.f; }, "B1");
  auto B2 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B1(i, j) + 1.f; }, "B2");

  auto C = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B(i, j) * 2.f; }, "C");
  auto C1 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B1(i, j) * 2.f; }, "C1");
  auto C2 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B2(i, j) * 2.f; }, "C2");

  auto stages = CreateStages({C, C1, C2});

  stages[B]->ComputeInline();
  stages[B1]->ComputeInline();
  stages[B2]->ComputeInline();

  auto fn = Lower("fn", stages, {A, C, C1, C2});

  LOG(INFO) << "fn:\n" << fn;

  auto target = R"ROC(
function fn (_A, _C, _C1, _C2)
{
  serial for (i, 0, 100)
  {
    serial for (j, 0, 200)
    {
      C2[i, j] = (6.00000000f + (2.00000000f * A[i, j]))
    }
  }
  serial for (i, 0, 100)
  {
    serial for (j, 0, 200)
    {
      C1[i, j] = (4.00000000f + (2.00000000f * A[i, j]))
    }
  }
  serial for (i, 0, 100)
  {
    serial for (j, 0, 200)
    {
      C[i, j] = (2.00000000f + (2.00000000f * A[i, j]))
    }
  }
}
  )ROC";

  ASSERT_EQ(utils::Trim(target), utils::Trim(utils::GetStreamCnt(fn)));
}

TEST(ShareBufferWith, basic) {
  Expr M(100), N(200);
  Placeholder<float> A("A", {M, N});

  auto B = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return A(i, j) + 1.f; }, "B");
  auto B1 = Compute(
      {M, N}, [=](Expr i, Expr j) -> Expr { return B(i, j) + 1.f; }, "B1");

  auto stages = CreateStages({B, B1});

  stages[B1]->ShareBufferWith(stages[B]);

  auto fn = Lower("fn", stages, {A, B, B1});

  LOG(INFO) << "fn:\n" << fn;

  Module::Builder builder("some_module", common::DefaultHostTarget());
  builder.AddFunction(fn);

  CodeGenC codegen(common::DefaultHostTarget());
  codegen.SetInlineBuiltinCodes(false);

  LOG(INFO) << "\n"
            << codegen.Compile(builder.Build(),
                               backends::CodeGenC::OutputKind::CImpl);
}

TEST(isl, test) {
  isl::ctx ctx(isl_ctx_alloc());
  isl::set domain(ctx,
                  "[p0, p1] -> { p[i, j] : p0 = 0 and 0 <= p1 <= 2 and 4p1 <= "
                  "i <= 1 + 4p1 and 0 <= j <= 9 + 4p1 - i }");

  isl::map schedule(ctx,
                    "[p0, p1] -> { p[i, j] -> p[t0, t1, t2 = j] : 2t1 = i and "
                    "(t0) mod 2 = 0 and 0 <= t0 <= 1 }");

  auto schedule_intersected = schedule.intersect_domain(domain);
  LOG(INFO) << "schedule_intersected: " << schedule_intersected.coalesce();

  isl::set context(ctx, "[p0,p1]->{:p0<100 and p1<100}");
  LOG(INFO) << "space: " << context.space();

  auto* build = isl_ast_build_from_context(context.release());
  auto* node = isl_ast_build_node_from_schedule_map(
      build, isl_union_map_from_map(schedule_intersected.release()));
  LOG(INFO) << "code:\n" << isl_ast_node_to_C_str(node);
}

TEST(isl, test1) {
  isl::ctx ctx(isl_ctx_alloc());

  isl::set domain(ctx,
                  "[p0, p1] -> { p[i, j] : p0 = 0 and 0 <= p1 <= 2 and 4p1 <= "
                  "i <= 1 + 4p1 and 0 <= j <= 9 + 4p1 - i }");
  isl::map schedule(ctx,
                    "[p0, p1] -> { p[i, j] -> p[o0, o1, t0, t1, t2 = j] : 2t1 "
                    "= i and (o0) mod 4 = 0 and (t0) mod 2 = 0 "
                    "and 0 <= o0 <= 3 and 0 <= o1 <= 2 and 0 <= t0 <= 1 }");

  isl::map schedule_t(ctx,
                      "[p0,p1] -> { p[i0,i1,i2,i3,i4] -> [t0,t1,t2,t3,t30,t4] "
                      ": t0 =i0 and t1 = i1 and t2 = i2 and t3 "
                      "= i3 and t4 = i4 and t30=0 }");

  isl::set cdomain(ctx, "[p0,p1] -> { c[a,b,c]: 0<=a,b,c<10 }");
  isl::map cschedule(ctx,
                     "[p0,p1] -> { c[a,b,c] -> c[t0,t1,t2,t3]: t0=a%4 and "
                     "t1=a/4 and t2=b and t3=c }");

  isl::map schedule_t1(ctx,
                       "[p0,p1] -> { c[i0,i1,i2,i3] -> [t0,t1,t2,t3,t30,t4] : "
                       "t0 =i0 and t1 = i1 and t2 = i2 and t3=i3 "
                       "and t4=0 and t30=1 }");

  schedule = schedule.apply_range(schedule_t);
  cschedule = cschedule.apply_range(schedule_t1);

  auto whole_domain = isl::manage(isl_union_set_from_set(domain.copy()));
  whole_domain = isl::manage(
      isl_union_set_add_set(whole_domain.release(), cdomain.copy()));

  auto whole_schedule = isl::manage(isl_union_map_from_map(schedule.copy()));
  whole_schedule = isl::manage(
      isl_union_map_add_map(whole_schedule.release(), cschedule.copy()));

  auto intersect_schedule = whole_schedule.intersect_domain(whole_domain);

  isl::set context(ctx, "[p0,p1]->{:p0<100 and p1<100}");

  auto* build = isl_ast_build_from_context(context.release());
  auto* node =
      isl_ast_build_node_from_schedule_map(build, intersect_schedule.release());
  LOG(INFO) << "code:\n\n" << isl_ast_node_to_C_str(node);
}

}  // namespace poly
}  // namespace cinn
