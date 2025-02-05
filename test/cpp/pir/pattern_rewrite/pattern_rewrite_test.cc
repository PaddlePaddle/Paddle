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

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/general/constant_folding_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/gpu/conv2d_add_act_fuse_pass.h"
#include "paddle/fluid/pir/transforms/gpu/conv2d_add_fuse_pass.h"
#include "paddle/fluid/pir/transforms/gpu/conv2d_bn_fuse_pass.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/cast_utils.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_info.h"
#include "paddle/pir/include/core/parameter.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "test/cpp/pir/tools/macros_utils.h"

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sqrt, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(divide, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(multiply, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(subtract, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(reshape, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(fetch, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(conv2d, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(transpose, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(cummax, CPU, ALL_LAYOUT);

// Define op1.
class Operation1 : public pir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "test.Operation1"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];  // NOLINT
  void VerifySig();
  static void InferShape() { VLOG(2) << "This is op2's InferShape interface."; }
};

void Operation1::VerifySig() {
  auto &attributes = this->attributes();
  if (attributes.count("op2_attr1") == 0 ||
      (!attributes.at("op2_attr1").isa<pir::StrAttribute>())) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Type of attribute: parameter_name is not right."));
  }
  if (attributes.count("op2_attr2") == 0 ||
      (!attributes.at("op2_attr2").isa<pir::StrAttribute>())) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Type of attribute: parameter_name is not right."));
  }
}
const char *Operation1::attributes_name[attributes_num] = {  // NOLINT
    "op2_attr1",
    "op2_attr2"};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(Operation1)
IR_DEFINE_EXPLICIT_TYPE_ID(Operation1)

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public pir::Dialect {
 public:
  explicit TestDialect(pir::IrContext *context)
      : pir::Dialect(name(), context, pir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "test"; }

 private:
  void initialize() { RegisterOps<Operation1>(); }
};
IR_DECLARE_EXPLICIT_TEST_TYPE_ID(TestDialect)
IR_DEFINE_EXPLICIT_TYPE_ID(TestDialect)

// TODO(wilber): Add logical when ir support erase, replace or update.
class TestPatternRewrite : public pir::OpRewritePattern<Operation1> {
 public:
  using pir::OpRewritePattern<Operation1>::OpRewritePattern;

  void Rewrite(Operation1 op, pir::PatternRewriter &rewriter) const override {}
  bool Match(Operation1 op) const override { return false; }
};

class TestPatternRewrite2 : public pir::OpRewritePattern<Operation1> {
 public:
  using pir::OpRewritePattern<Operation1>::OpRewritePattern;
  bool MatchAndRewrite(
      Operation1 op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    return false;
  }
};

TEST(PatternRewrite, PatternBenefit) {
  pir::PatternBenefit benefit1(1);
  EXPECT_EQ(benefit1.benefit(), 1U);
  pir::PatternBenefit benefit2(2);
  EXPECT_EQ(benefit2.benefit(), 2U);

  EXPECT_TRUE(benefit2 > benefit1);
  EXPECT_TRUE(benefit2 >= benefit1);
  EXPECT_TRUE(benefit1 < benefit2);
  EXPECT_TRUE(benefit1 <= benefit2);
  EXPECT_TRUE(benefit1 != benefit2);
  pir::PatternBenefit benefit3(2);
  EXPECT_TRUE(benefit2 == benefit3);
}

TEST(RewritePattern, RewritePatternSet) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  test_dialect->RegisterOp<Operation1>();

  pir::RewritePatternSet ps(ctx);
  ps.Add<TestPatternRewrite>(ctx, 1);
  EXPECT_EQ(ps.native_patterns().size(), 1U);
  EXPECT_TRUE(ps.native_patterns().back()->debug_labels().empty());
  EXPECT_EQ(ps.native_patterns().back()->benefit(), 1U);
  ps.AddWithLabel<TestPatternRewrite2>({"TestPatternRewrite2"}, ctx, 2);
  EXPECT_EQ(ps.native_patterns().size(), 2U);
  EXPECT_EQ(ps.native_patterns().back()->debug_labels()[0],
            "TestPatternRewrite2");
  EXPECT_EQ(ps.native_patterns().back()->benefit(), 2U);

  ps.Clear();
  ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);
  EXPECT_EQ(ps.native_patterns().size(), 2U);
  EXPECT_EQ(ps.native_patterns()[0]->benefit(), 2U);
  EXPECT_EQ(ps.native_patterns()[1]->benefit(), 2U);
}

// TODO(wilber): Add actual case.
// TEST(PatternRewrite, PatternApplicator) {
//   pir::IrContext *ctx = pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
//   auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
//   test_dialect->RegisterOp<Operation1>();
//   pir::RewritePatternSet ps(ctx);
//   ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);
//   pir::FrozenRewritePatternSet frozen_set(std::move(ps));
//   pir::PatternApplicator applicator(frozen_set);
//   applicator.ApplyDefaultCostModel();
// }

// // TODO(wilber): Add actual case.
TEST(PatternRewrite, FrozenRewritePatternSet) {
  pir::FrozenRewritePatternSet frozen_set;
  EXPECT_TRUE(frozen_set.match_any_op_native_patterns().empty());
  EXPECT_TRUE(frozen_set.op_specific_native_patterns().empty());

  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  test_dialect->RegisterOp<Operation1>();
  pir::RewritePatternSet ps(ctx);
  ps.Add<TestPatternRewrite, TestPatternRewrite2>(ctx, 2);

  pir::FrozenRewritePatternSet frozen_set2(std::move(ps));
  EXPECT_TRUE(frozen_set2.match_any_op_native_patterns().empty());
  const auto &pattern_maps = frozen_set2.op_specific_native_patterns();
  EXPECT_EQ(pattern_maps.size(), 1U);
  EXPECT_EQ(pattern_maps.at(ctx->GetRegisteredOpInfo("test.Operation1")).size(),
            2U);
}

class RedundantTransposeFusePattern
    : public pir::OpRewritePattern<paddle::dialect::TransposeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::TransposeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::TransposeOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto prev_op = pir::GetDefiningOpForInput(op, 0);
    std::vector<int> axis_last = GetAxis(op);
    auto prev_trans_op = prev_op->dyn_cast<paddle::dialect::TransposeOp>();
    if (prev_trans_op) {
      std::vector<int> axis_first = GetAxis(prev_trans_op);
      PADDLE_ENFORCE_EQ(axis_first.size(),
                        axis_last.size(),
                        common::errors::InvalidArgument(
                            "transpose op's perm rank should be same."));
      auto new_perm = GetPerm(axis_first, axis_last);
      rewriter.set_insertion_point(op);
      auto new_transpose_op = rewriter.Build<paddle::dialect::TransposeOp>(
          pir::GetDefiningOpForInput(prev_trans_op, 0)->result(0), new_perm);
      rewriter.ReplaceOp(op, {new_transpose_op.out()});
      return true;
    }

    return false;
  }

 private:
  std::vector<int> GetAxis(paddle::dialect::TransposeOp op) const {
    auto array_attr = op.attribute<pir::ArrayAttribute>("perm").AsVector();
    std::vector<int> axis(array_attr.size());
    for (size_t i = 0; i < array_attr.size(); ++i) {
      axis[i] = array_attr[i].dyn_cast<pir::Int32Attribute>().data();
    }
    return axis;
  }

  std::vector<int> GetPerm(const std::vector<int> &perm1,
                           const std::vector<int> &perm2) const {
    int n = static_cast<int>(perm1.size());
    std::vector<int> axis(n), axis1(n), axis2(n);
    std::iota(axis.begin(), axis.end(), 0);
    for (int i = 0; i < n; ++i) {
      axis1[i] = axis[perm1[i]];
    }
    for (int i = 0; i < n; ++i) {
      axis2[i] = axis1[perm2[i]];
    }
    return axis2;
  }
};

class TestPass : public pir::PatternRewritePass {
 public:
  TestPass() : pir::PatternRewritePass("test_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<RedundantTransposeFusePattern>(context);
    return ps;
  }
};

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_filter_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 3, 3, 3},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_input_op_1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_filter_op_1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 3, 3, 3},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp add_op_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 64, 1, 1},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_input_op_2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 3, 16, 16},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_filter_op_2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 3, 3, 3},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp add_op_y_1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 64, 1, 1},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp add_op_2_x = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{4, 64, 14, 14},
      1.5,
      phi::DataType::FLOAT32,
      phi::CPUPlace());

  paddle::dialect::FullOp full_mean_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{64}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::FullOp full_variance_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_scale_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::FullOp full_bias_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{64}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::Conv2dOp conv2d_op =
      builder.Build<paddle::dialect::Conv2dOp>(full_input_op.out(),
                                               full_filter_op.out());

  paddle::dialect::BatchNorm_Op batch_norm_op =
      builder.Build<paddle::dialect::BatchNorm_Op>(conv2d_op.out(),
                                                   full_mean_op.out(),
                                                   full_variance_op.out(),
                                                   full_scale_op.out(),
                                                   full_bias_op.out(),
                                                   true,
                                                   0.9,
                                                   1e-6,
                                                   "NCHW",
                                                   false,
                                                   false);
  paddle::dialect::Conv2dOp conv2d_op_1 =
      builder.Build<paddle::dialect::Conv2dOp>(full_input_op_1.out(),
                                               full_filter_op_1.out());
  paddle::dialect::AddOp add_op =
      builder.Build<paddle::dialect::AddOp>(conv2d_op_1.out(), add_op_y.out());
  paddle::dialect::ReluOp relu_op =
      builder.Build<paddle::dialect::ReluOp>(add_op.out());

  paddle::dialect::Conv2dOp conv2d_op_2 =
      builder.Build<paddle::dialect::Conv2dOp>(full_input_op_2.out(),
                                               full_filter_op_2.out());
  paddle::dialect::AddOp add_op_1 = builder.Build<paddle::dialect::AddOp>(
      conv2d_op_2.out(), add_op_y_1.out());
  paddle::dialect::AddOp add_op_2 =
      builder.Build<paddle::dialect::AddOp>(add_op_2_x.out(), add_op_1.out());
  paddle::dialect::ReluOp relu_op_1 =
      builder.Build<paddle::dialect::ReluOp>(add_op_2.out());

  auto transpose1_op = builder.Build<paddle::dialect::TransposeOp>(
      batch_norm_op.out(), std::vector<int>{0, 2, 3, 1});

  auto transpose2_op = builder.Build<paddle::dialect::TransposeOp>(
      transpose1_op.out(), std::vector<int>{0, 3, 1, 2});
  auto add_out = builder.Build<paddle::dialect::AddOp>(transpose2_op.out(),
                                                       relu_op_1.out());
  auto add_out_1 =
      builder.Build<paddle::dialect::AddOp>(add_out.out(), relu_op.out());
  builder.Build<paddle::dialect::FetchOp>(add_out_1.out(), "out", 0);
}

TEST(pattern_rewrite, Patterns) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 27u);

  pir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<TestPass>());
  pm.AddPass(pir::CreateConv2dBnFusePass());
  pm.AddPass(pir::CreateConv2dAddActFusePass());
  pm.AddPass(pir::CreateConv2dAddFusePass());
  std::unique_ptr<pir::Pass> constant_folding_pass =
      pir::CreateConstantFoldingPass();
  phi::Place place = phi::CPUPlace();
  constant_folding_pass->SetNotOwned(pir::Pass::kPlaceAttr, &place);
  constant_folding_pass->Set(pir::Pass::kParamScopeAttr,
                             new paddle::framework::Scope());
  pm.AddPass(std::move(constant_folding_pass));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  // pm.EnablePassTiming();
  pm.EnableIRPrinting();
  // pm.EnableIRPrinting(std::make_unique<pir::PassManager::IRPrinterOption>(
  //     [](pir::Pass *pass, pir::Operation *op) {
  //       return pass->name() == "constant_folding_pass";
  //     },
  //     [](pir::Pass *pass, pir::Operation *op) {
  //       return pass->name() == "constant_folding_pass";
  //     },
  //     true,
  //     true));
  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::ExecutionTimeout(
                        "Sorry, the for program test is timeout."));
  EXPECT_EQ(program.block()->size(), 17u);
}

void BuildConstantFoldingProgram(pir::Program *program,
                                 pir::IrContext *ctx,
                                 paddle::framework::Scope *scope) {
  pir::Builder builder = pir::Builder(ctx, program->block());

  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {2, 2};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0, 1, 2}};
  size_t offset = 0;
  pir::Type dense_tensor_dtype = paddle::dialect::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  phi::DenseTensorMeta meta(
      phi::DataType::FLOAT32, dims, data_layout, lod, offset);
  phi::DeviceContext *dev_ctx =
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace());

  auto op1 = builder.Build<pir::ConstantTensorOp>("a", dense_tensor_dtype);
  auto op2 = builder.Build<pir::ConstantTensorOp>("b", dense_tensor_dtype);

  auto op3 =
      builder.Build<paddle::dialect::AddOp>(op1->result(0), op2->result(0));

  auto op4 = builder.Build<pir::ParameterOp>("c", dense_tensor_dtype);

  auto op5 =
      builder.Build<paddle::dialect::AddOp>(op3->result(0), op4->result(0));
  builder.Build<paddle::dialect::FetchOp>(op5.out(), "out", 0);

  auto *tensor_a = scope->Var("a")->GetMutable<phi::DenseTensor>();
  auto *tensor_b = scope->Var("b")->GetMutable<phi::DenseTensor>();
  auto *tensor_c = scope->Var("c")->GetMutable<phi::DenseTensor>();

  tensor_a->set_meta(meta);
  tensor_b->set_meta(meta);
  tensor_c->set_meta(meta);

  dev_ctx->Alloc(tensor_a, phi::DataType::FLOAT32);
  dev_ctx->Alloc(tensor_b, phi::DataType::FLOAT32);
  dev_ctx->Alloc(tensor_c, phi::DataType::FLOAT32);
}

TEST(constant_folding, ConstantFolding) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Program program(ctx);
  paddle::framework::Scope scope;
  BuildConstantFoldingProgram(&program, ctx, &scope);

  pir::PassManager pm(ctx);
  std::unique_ptr<pir::Pass> constant_folding_pass =
      pir::CreateConstantFoldingPass();
  phi::Place place = phi::CPUPlace();
  constant_folding_pass->SetNotOwned(pir::Pass::kPlaceAttr, &place);
  constant_folding_pass->SetNotOwned(pir::Pass::kParamScopeAttr, &scope);
  pm.AddPass(std::move(constant_folding_pass));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::ExecutionTimeout(
                        "Sorry, the test for program is timeout."));
  EXPECT_EQ(program.block()->size(), 2u);
}

TEST(constant_folding, ConstantFolding_Train) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Program program(ctx);
  paddle::framework::Scope scope;
  BuildConstantFoldingProgram(&program, ctx, &scope);

  pir::PassManager pm(ctx);
  std::unique_ptr<pir::Pass> constant_folding_pass =
      pir::CreateConstantFoldingPass();
  phi::Place place = phi::CPUPlace();
  constant_folding_pass->SetNotOwned(pir::Pass::kPlaceAttr, &place);
  constant_folding_pass->SetNotOwned(pir::Pass::kParamScopeAttr, &scope);
  constant_folding_pass->Set("train_mode", new bool(true));

  pm.AddPass(std::move(constant_folding_pass));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::ExecutionTimeout(
                        "Sorry, the test for program is timeout."));
  EXPECT_EQ(program.block()->size(), 4u);
}

void BuildConcatProgram(pir::Program *program, pir::IrContext *ctx) {
  pir::Builder builder = pir::Builder(ctx, program->block());
  auto x = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto y = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto t1 =
      builder.Build<pir::CombineOp>(std::vector<pir::Value>({x, y})).result(0);

  auto out1 = builder.Build<paddle::dialect::ConcatOp>(t1, 1).result(0);

  auto z = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto w = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({16, 16}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto t2 =
      builder.Build<pir::CombineOp>(std::vector<pir::Value>({z, w})).result(0);

  auto out2 = builder.Build<paddle::dialect::ConcatOp>(t2, 1).result(0);

  auto out = builder.Build<paddle::dialect::AddOp>(out1, out2).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
}

TEST(constant_folding, ConstantFolding_Combine) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Program program(ctx);
  BuildConcatProgram(&program, ctx);

  pir::PassManager pm(ctx);
  std::unique_ptr<pir::Pass> constant_folding_pass =
      pir::CreateConstantFoldingPass();
  phi::Place place = phi::CPUPlace();
  constant_folding_pass->SetNotOwned(pir::Pass::kPlaceAttr, &place);
  constant_folding_pass->Set(pir::Pass::kParamScopeAttr,
                             new paddle::framework::Scope());
  pm.AddPass(std::move(constant_folding_pass));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::ExecutionTimeout(
                        "Sorry, the test for program is timeout."));
  EXPECT_EQ(program.block()->size(), 2u);
}

void BuildMultiOutputProgram(pir::Program *program, pir::IrContext *ctx) {
  pir::Builder builder = pir::Builder(ctx, program->block());
  auto x = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({2, 2}),
                                               2.0,
                                               phi::DataType::FLOAT32,
                                               phi::GPUPlace())
               .result(0);

  auto cummax_op = builder.Build<paddle::dialect::CummaxOp>(x, 0);

  auto out = cummax_op.out();
  auto indices = cummax_op.indices();

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  builder.Build<paddle::dialect::FetchOp>(indices, "indices", 1);
}

TEST(constant_folding, ConstantFolding_MultiOutput) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  pir::Program program(ctx);
  BuildMultiOutputProgram(&program, ctx);

  pir::PassManager pm(ctx);
  std::unique_ptr<pir::Pass> constant_folding_pass =
      pir::CreateConstantFoldingPass();
  phi::Place place = phi::CPUPlace();
  constant_folding_pass->SetNotOwned(pir::Pass::kPlaceAttr, &place);
  constant_folding_pass->Set(pir::Pass::kParamScopeAttr,
                             new paddle::framework::Scope());
  pm.AddPass(std::move(constant_folding_pass));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::ExecutionTimeout(
                        "Sorry, the test for program is timeout."));
  EXPECT_EQ(program.block()->size(), 4u);
}
