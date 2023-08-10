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

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"

#include "paddle/fluid/ir/transforms/fusion_merge_pass.h"

std::vector<ir::OpResult> BuildInput(
    ::ir::Builder* builder,
    const std::vector<std::vector<int64_t>>& vec_shapes) {
  std::vector<ir::OpResult> vec_res;
  for (size_t i = 0; i < vec_shapes.size(); ++i) {
    auto op = builder->Build<paddle::dialect::FullOp>(
        vec_shapes[i], 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

    vec_res.push_back(op.result(0));
  }

  return vec_res;
}

TEST(IROpFusionPass, demo) {
  ::ir::IrContext* ctx = ::ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ::ir::Program program_base(ctx);
  ::ir::Builder builder_base = ::ir::Builder(ctx, program_base.block());

  auto inputs = BuildInput(&builder_base, {{10, 10}, {10, 10}});

  ::ir::Program program(ctx);
  ::ir::Builder builder = ::ir::Builder(ctx, program.block());

  auto add = builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]);
  builder.Build<paddle::dialect::ReluOp>(add.result(0));

  program.Print(std::cout);
  auto res = ::ir::OpFusionPassInternal(program);

  ASSERT_EQ(res.size(), 1u);
}

TEST(IROpFusionPass, ElementWise_Fusion_0) {
  ::ir::IrContext* ctx = ::ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ::ir::Program program_base(ctx);
  ::ir::Builder builder_base = ::ir::Builder(ctx, program_base.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder_base, {{h, w}, {h, w}, {h, w}, {h, w}});

  ::ir::Program program(ctx);
  ::ir::Builder builder = ::ir::Builder(ctx, program.block());

  auto add1 = builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]);
  auto add2 = builder.Build<paddle::dialect::AddOp>(inputs[2], inputs[3]);
  builder.Build<paddle::dialect::AddOp>(add1.result(0), add2.result(0));

  program.Print(std::cout);
  auto res = ::ir::OpFusionPassInternal(program);

  ASSERT_EQ(res.size(), 1u);
}

TEST(IROpFusionPass, ElementWise_Fusion_1) {
  ::ir::IrContext* ctx = ::ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ::ir::Program program_base(ctx);
  ::ir::Builder builder_base = ::ir::Builder(ctx, program_base.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder_base, {{h, w}, {h, w}, {h, w}, {h, w}});

  ::ir::Program program(ctx);
  ::ir::Builder builder = ::ir::Builder(ctx, program.block());

  auto e =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto f = builder.Build<paddle::dialect::AddOp>(e, inputs[2]).result(0);
  auto g = builder.Build<paddle::dialect::AddOp>(e, inputs[3]).result(0);
  builder.Build<paddle::dialect::AddOp>(f, g);

  program.Print(std::cout);
  auto res = ::ir::OpFusionPassInternal(program);

  ASSERT_EQ(res.size(), 1u);
}

// TEST(IROpFusionPass, Brodcast_Test_0) {
//   ::ir::IrContext* ctx = ::ir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
//   ::ir::Program program_base(ctx);
//   ::ir::Builder builder_base = ::ir::Builder(ctx, program_base.block());

//   int h = 32, w = 32;
//   auto inputs = BuildInput(&builder_base, { {w}, {w}, {h, w}, {h, w}});

//   ::ir::Program program(ctx);
//   ::ir::Builder builder = ::ir::Builder(ctx, program.block());

//   auto e =  builder.Build<paddle::dialect::AddOp>( inputs[0], inputs[1]
//   ).result(0); auto f = builder.Build<paddle::dialect::AddOp>( e, inputs[2]
//   ).result(0); auto g = builder.Build<paddle::dialect::AddOp>( e,
//   inputs[3]).result(0); builder.Build<paddle::dialect::AddOp>( f, g);

//   program.Print( std::cout );
//   auto res = ::ir::OpFusionPassInternal( program );

//   ASSERT_EQ( res.size(), 1u);
// }

// TEST(IROpFusionPass, Reduce_Test_0) {
//   ::ir::IrContext* ctx = ::ir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
//   ::ir::Program program_base(ctx);
//   ::ir::Builder builder_base = ::ir::Builder(ctx, program_base.block());

//   int h = 32, w = 32;
//   auto inputs = BuildInput(&builder_base, {{w}, {w}, {h, w}, {h, w}});

//   ::ir::Program program(ctx);
//   ::ir::Builder builder = ::ir::Builder(ctx, program.block());

//   auto e =
//       builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
//   auto f =
//       builder.Build<paddle::dialect::AddOp>(inputs[2], inputs[3]).result(0);
//   auto g =
//       builder
//           .Build<paddle::dialect::SumOp>(f, {0}, phi::DataType::FLOAT32,
//           false) .result(0);
//   builder.Build<paddle::dialect::AddOp>(e, g);

//   program.Print(std::cout);
//   auto res = ::ir::OpFusionPassInternal(program);

//   ASSERT_EQ(res.size(), 1u);
// }

// TEST(IROpFusionPass, Reduce_Test_1) {
//   ::ir::IrContext* ctx = ::ir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
//   ::ir::Program program_base(ctx);
//   ::ir::Builder builder_base = ::ir::Builder(ctx, program_base.block());

//   int h = 32, w = 32;
//   auto inputs = BuildInput(&builder_base, {{w}, {w}, {h, w}, {h, w}});

//   ::ir::Program program(ctx);
//   ::ir::Builder builder = ::ir::Builder(ctx, program.block());

//   auto e =
//       builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
//   auto f = builder
//                .Build<paddle::dialect::SumOp>(
//                    inputs[2], {0}, phi::DataType::FLOAT32, false)
//                .result(0);
//   auto g = builder
//                .Build<paddle::dialect::SumOp>(
//                    inputs[3], {0}, phi::DataType::FLOAT32, false)
//                .result(0);
//   auto h = builder.Build<paddle::dialect::AddOp>(e, f).result(0);
//   builder.Build<paddle::dialect::AddOp>(g, h);

//   program.Print(std::cout);
//   auto res = ::ir::OpFusionPassInternal(program);

//   ASSERT_EQ(res.size(), 1u);
// }

// TEST(IROpFusionPass, Reduce_Test_2) {
//   ::ir::IrContext* ctx = ::ir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
//   ::ir::Program program_base(ctx);
//   ::ir::Builder builder_base = ::ir::Builder(ctx, program_base.block());

//   int h = 32, w = 32;
//   auto inputs = BuildInput(&builder_base, {{w}, {w}, {h, w}, {h, w}});

//   ::ir::Program program(ctx);
//   ::ir::Builder builder = ::ir::Builder(ctx, program.block());

//   auto e = builder
//                .Build<paddle::dialect::SumOp>(
//                    inputs[2], {0}, phi::DataType::FLOAT32, false)
//                .result(0);
//   auto f = builder
//                .Build<paddle::dialect::SumOp>(
//                    inputs[3], {0}, phi::DataType::FLOAT32, false)
//                .result(0);
//   auto g = builder.Build<paddle::dialect::AddOp>(inputs[0], e).result(0);
//   auto h = builder.Build<paddle::dialect::AddOp>(inputs[1], f).result(0);
//   builder.Build<paddle::dialect::AddOp>(g, h);

//   program.Print(std::cout);
//   auto res = ::ir::OpFusionPassInternal(program);

//   ASSERT_EQ(res.size(), 2u);
// }
