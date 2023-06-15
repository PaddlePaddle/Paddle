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

#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"

TEST(PatternBenefit, PatternBenefit) {
  ir::PatternBenefit benefit1(1);
  EXPECT_EQ(benefit1.benefit(), 1U);
  ir::PatternBenefit benefit2(2);
  EXPECT_EQ(benefit2.benefit(), 2U);

  EXPECT_TRUE(benefit2 > benefit1);
  EXPECT_TRUE(benefit2 >= benefit1);
  EXPECT_TRUE(benefit1 < benefit2);
  EXPECT_TRUE(benefit1 <= benefit2);
  EXPECT_TRUE(benefit1 != benefit2);
  ir::PatternBenefit benefit3(2);
  EXPECT_TRUE(benefit2 == benefit3);
}

// Define op1.
class Operation1 : public ir::Op<Operation1> {
 public:
  using Op::Op;
  static const char *name() { return "test.Operation1"; }
  static constexpr uint32_t attributes_num = 2;
  static const char *attributes_name[attributes_num];
  static void Verify(const std::vector<ir::OpResult> &inputs,
                     const std::vector<ir::Type> &outputs,
                     const ir::AttributeMap &attributes) {
    if (attributes.count("op2_attr1") == 0 ||
        (!attributes.at("op2_attr1").isa<ir::StrAttribute>())) {
      throw("Type of attribute: parameter_name is not right.");
    }
    if (attributes.count("op2_attr2") == 0 ||
        (!attributes.at("op2_attr2").isa<ir::StrAttribute>())) {
      throw("Type of attribute: parameter_name is not right.");
    }
  }
  static void InferShape() { VLOG(2) << "This is op2's InferShape interface."; }
};
const char *Operation1::attributes_name[attributes_num] = {"op2_attr1",
                                                           "op2_attr2"};

// Define a dialect, op1 and op2 will be registered by this dialect.
class TestDialect : public ir::Dialect {
 public:
  explicit TestDialect(ir::IrContext *context)
      : ir::Dialect(name(), context, ir::TypeId::get<TestDialect>()) {
    initialize();
  }
  static const char *name() { return "test"; }

 private:
  void initialize() { RegisterOps<Operation1>(); }
};

// TODO(wilber): Add logical when ir support erase, replace or update.
class TestPatternRewrite : public ir::OpRewritePattern<Operation1> {
 public:
  using ir::OpRewritePattern<Operation1>::OpRewritePattern;

  void Rewrite(Operation1 op, ir::PatternRewriter &rewriter) const override {}
  bool Match(Operation1 op) const override { return false; }
};

class TestPatternRewrite2 : public ir::OpRewritePattern<Operation1> {
 public:
  using ir::OpRewritePattern<Operation1>::OpRewritePattern;
  bool MatchAndRewrite(
      Operation1 op,
      ir::PatternRewriter &rewriter) const override {  // NOLINT
    return false;
  }
};

TEST(RewritePattern, OpRewritePattern) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  auto *test_dialect = ctx->GetOrRegisterDialect<TestDialect>();
  test_dialect->RegisterOp<Operation1>();

  ir::RewritePatternSet ps(ctx);
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
