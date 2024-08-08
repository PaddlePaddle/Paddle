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

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

std::vector<pir::Value> BuildInput(
    ::pir::Builder* builder,
    const std::vector<std::vector<int64_t>>& vec_shapes) {
  std::vector<pir::Value> vec_res;
  for (size_t i = 0; i < vec_shapes.size(); ++i) {
    auto op = builder->Build<paddle::dialect::FullOp>(
        vec_shapes[i], 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

    vec_res.push_back(op.result(0));
  }

  return vec_res;
}

TEST(IROpFusionPass, demo) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  auto x = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({10, 10}),
                                               1.0,
                                               phi::DataType::FLOAT32,
                                               phi::CPUPlace())
               .result(0);
  auto y = builder
               .Build<paddle::dialect::FullOp>(std::vector<int64_t>({10, 10}),
                                               1.0,
                                               phi::DataType::FLOAT32,
                                               phi::CPUPlace())
               .result(0);
  auto add = builder.Build<paddle::dialect::AddOp>(x, y).result(0);

  auto sum = builder
                 .Build<cinn::dialect::ReduceSumOp>(
                     add, std::vector<int64_t>({-1}), true)
                 .result(0);

  auto out1 = builder.Build<paddle::dialect::ReluOp>(sum).result(0);
  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  ASSERT_EQ(program.block()->size(), 2u);
}

TEST(IROpFusionPass, ElementWise_Fusion_0) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{h, w}, {h, w}, {h, w}});

  auto e =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto f = builder.Build<paddle::dialect::AddOp>(e, inputs[2]).result(0);
  auto out1 = builder.Build<paddle::dialect::AddOp>(f, inputs[2]).result(0);
  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));
  ASSERT_EQ(program.block()->size(), 2u);
}

// Real test 0
TEST(IROpFusionPass, Broadcast_Test_0) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{w}, {w}, {h, w}, {h, w}});

  auto e =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto f =
      builder.Build<paddle::dialect::AddOp>(inputs[2], inputs[3]).result(0);
  std::vector<int64_t> axes{1};
  std::vector<int64_t> out_shape{h, w};
  auto e1 =
      builder.Build<cinn::dialect::BroadcastOp>(e, axes, out_shape).result(0);
  auto out1 = builder.Build<paddle::dialect::AddOp>(e1, f).result(0);
  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  ASSERT_EQ(program.block()->size(), 2u);
}

// Real test 1
TEST(IROpFusionPass, Broadcast_Test_1) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{w}, {w}, {w}, {h, w}});

  auto e =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto out1 = builder.Build<paddle::dialect::AddOp>(inputs[2], e).result(0);
  std::vector<int64_t> axes{1};
  std::vector<int64_t> out_shape{h, w};
  auto e1 =
      builder.Build<cinn::dialect::BroadcastOp>(e, axes, out_shape).result(0);
  auto out2 = builder.Build<paddle::dialect::AddOp>(inputs[3], e1).result(0);

  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  ASSERT_EQ(program.block()->size(), 4u);
}

TEST(IROpFusionPass, Broadcast_Test_2) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{w}, {w}, {w}, {h, w}, {h, w}});

  auto f =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto out1 = builder.Build<paddle::dialect::AddOp>(inputs[2], f).result(0);
  std::vector<int64_t> axes{1};
  std::vector<int64_t> out_shape{h, w};
  auto f1 =
      builder.Build<cinn::dialect::BroadcastOp>(f, axes, out_shape).result(0);
  auto out2 = builder.Build<paddle::dialect::AddOp>(inputs[3], f1).result(0);
  auto out3 = builder.Build<paddle::dialect::AddOp>(inputs[4], f1).result(0);

  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);
  builder.Build<paddle::dialect::FetchOp>(out3, "out3", 2);
  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 5u
  ASSERT_EQ(program.block()->size(), 6u);
}

// Real reduce 0
TEST(IROpFusionPass, reduce_test_0) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{h, w}, {h, w}});

  std::vector<int64_t> axes{0};
  auto c =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto out1 =
      builder.Build<cinn::dialect::ReduceSumOp>(c, axes, true).result(0);
  auto out2 =
      builder.Build<cinn::dialect::ReduceSumOp>(c, axes, true).result(0);
  auto out3 =
      builder.Build<cinn::dialect::ReduceSumOp>(c, axes, true).result(0);

  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);
  builder.Build<paddle::dialect::FetchOp>(out3, "out3", 2);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 4u
  ASSERT_EQ(program.block()->size(), 6u);
}

// Real reduce 1
TEST(IROpFusionPass, reduce_test_1) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{h, w}, {h, w}});

  std::vector<int64_t> axes{0};
  std::vector<int64_t> axes1{1};
  auto c =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto out1 =
      builder.Build<cinn::dialect::ReduceSumOp>(c, axes, true).result(0);
  auto out2 =
      builder.Build<cinn::dialect::ReduceSumOp>(c, axes1, true).result(0);

  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 3u
  ASSERT_EQ(program.block()->size(), 4u);
}

// Real reduce 2
TEST(IROpFusionPass, reduce_test_2) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{h, w}, {h, w}, {w}});

  std::vector<int64_t> axes{0};
  std::vector<int64_t> axes1{1};
  auto d =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto e = builder.Build<cinn::dialect::ReduceSumOp>(d, axes, false).result(0);
  auto f = builder.Build<cinn::dialect::ReduceSumOp>(d, axes1, false).result(0);
  auto out1 = builder.Build<paddle::dialect::AddOp>(inputs[2], e).result(0);
  auto out2 = builder.Build<paddle::dialect::AddOp>(inputs[2], f).result(0);
  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 3u
  ASSERT_EQ(program.block()->size(), 6u);
}

// Real reduce 3
TEST(IROpFusionPass, reduce_test_3) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{h, w}, {h, w}, {w}});

  std::vector<int64_t> axes{0};
  std::vector<int64_t> axes1{1};
  auto e =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto f = builder.Build<cinn::dialect::ReduceSumOp>(e, axes, false).result(0);

  auto out1 = builder.Build<paddle::dialect::AddOp>(inputs[2], f).result(0);

  std::vector<int64_t> out_shape{h, w};
  auto f1 =
      builder.Build<cinn::dialect::BroadcastOp>(f, axes1, out_shape).result(0);
  auto out2 = builder.Build<paddle::dialect::AddOp>(inputs[2], f1).result(0);
  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 3u
  ASSERT_EQ(program.block()->size(), 6u);
}

TEST(IROpFusionPass, reduce_test_4) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{h, w}, {h, w}, {w}, {h, w}});

  std::vector<int64_t> axes{0};
  std::vector<int64_t> axes1{1};
  auto e =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto f = builder.Build<cinn::dialect::ReduceSumOp>(e, axes, false).result(0);

  auto out1 = builder.Build<paddle::dialect::AddOp>(inputs[2], f).result(0);

  std::vector<int64_t> out_shape{h, w};
  auto f1 =
      builder.Build<cinn::dialect::BroadcastOp>(f, axes1, out_shape).result(0);
  auto out2 = builder.Build<paddle::dialect::AddOp>(inputs[3], f1).result(0);
  auto f2 =
      builder.Build<cinn::dialect::BroadcastOp>(f, axes1, out_shape).result(0);
  auto out3 = builder.Build<paddle::dialect::AddOp>(inputs[3], f2).result(0);

  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);
  builder.Build<paddle::dialect::FetchOp>(out3, "out3", 2);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 4u
  ASSERT_EQ(program.block()->size(), 7u);
}

// Real reduce 5
TEST(IROpFusionPass, reduce_test_5) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder, {{h, w}, {h, w}});

  std::vector<int64_t> axes{1};

  auto c =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto out1 = builder.Build<cinn::dialect::ReduceSumOp>(inputs[0], axes, false)
                  .result(0);
  auto out2 = builder.Build<cinn::dialect::ReduceSumOp>(inputs[1], axes, false)
                  .result(0);
  auto out3 =
      builder.Build<cinn::dialect::ReduceSumOp>(c, axes, false).result(0);

  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);
  builder.Build<paddle::dialect::FetchOp>(out3, "out3", 2);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 4u
  ASSERT_EQ(program.block()->size(), 6u);
}

TEST(IROpFusionPass, layer_norm) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  auto inputs = BuildInput(&builder, {{128, 128, 768}, {768}, {768}});

  std::vector<int64_t> axes{-1};

  auto num = builder
                 .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
                                                 768.0,
                                                 phi::DataType::FLOAT32,
                                                 phi::CPUPlace())
                 .result(0);
  auto eps = builder
                 .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
                                                 1e-5,
                                                 phi::DataType::FLOAT32,
                                                 phi::CPUPlace())
                 .result(0);

  auto sum = builder.Build<cinn::dialect::ReduceSumOp>(inputs[0], axes, true)
                 .result(0);
  std::vector<int64_t> all_axes{0, 1, 2};
  std::vector<int64_t> out_shape1{128, 128, 1};
  auto num1 =
      builder.Build<cinn::dialect::BroadcastOp>(num, all_axes, out_shape1)
          .result(0);
  auto mean = builder.Build<paddle::dialect::DivideOp>(sum, num1).result(0);
  auto power = builder.Build<paddle::dialect::MultiplyOp>(inputs[0], inputs[0])
                   .result(0);
  auto power_sum =
      builder.Build<cinn::dialect::ReduceSumOp>(power, axes, true).result(0);
  auto mean2 =
      builder.Build<paddle::dialect::DivideOp>(power_sum, num1).result(0);
  auto power_mean =
      builder.Build<paddle::dialect::MultiplyOp>(mean, mean).result(0);

  auto var =
      builder.Build<paddle::dialect::SubtractOp>(mean2, power_mean).result(0);

  std::vector<int64_t> out_shape2{128, 128, 768};
  auto sub =
      builder.Build<paddle::dialect::SubtractOp>(inputs[0], mean).result(0);
  auto eps1 =
      builder.Build<cinn::dialect::BroadcastOp>(eps, all_axes, out_shape2)
          .result(0);
  auto t1 = builder.Build<paddle::dialect::AddOp>(var, eps1).result(0);
  auto t2 = builder.Build<paddle::dialect::SqrtOp>(t1).result(0);
  auto t3 = builder.Build<paddle::dialect::DivideOp>(sub, t2).result(0);
  auto scale =
      builder.Build<cinn::dialect::BroadcastOp>(inputs[1], all_axes, out_shape2)
          .result(0);
  auto bias =
      builder.Build<cinn::dialect::BroadcastOp>(inputs[2], all_axes, out_shape2)
          .result(0);
  auto t5 = builder.Build<paddle::dialect::MultiplyOp>(t3, scale).result(0);
  auto out1 = builder.Build<paddle::dialect::MultiplyOp>(t5, bias).result(0);

  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 2u
  ASSERT_EQ(program.block()->size(), 6u);
}

TEST(IROpFusionPass, softmax) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  auto inputs = BuildInput(&builder, {{128, 128, 768}});

  std::vector<int64_t> axes{-1};

  auto x = inputs[0];
  auto max = builder.Build<cinn::dialect::ReduceMaxOp>(x, axes, true).result(0);
  auto broadcast_1 = builder
                         .Build<cinn::dialect::BroadcastOp>(
                             max,
                             std::vector<int64_t>({0, 1, 2}),
                             std::vector<int64_t>({128, 128, 768}))
                         .result(0);
  auto sub =
      builder.Build<paddle::dialect::SubtractOp>(x, broadcast_1).result(0);
  auto exp = builder.Build<paddle::dialect::ExpOp>(sub).result(0);
  auto sum =
      builder.Build<cinn::dialect::ReduceSumOp>(exp, axes, true).result(0);

  auto broadcast_2 = builder
                         .Build<cinn::dialect::BroadcastOp>(
                             sum,
                             std::vector<int64_t>({0, 1, 2}),
                             std::vector<int64_t>({128, 128, 768}))
                         .result(0);
  auto divide =
      builder.Build<paddle::dialect::DivideOp>(exp, broadcast_2).result(0);

  builder.Build<paddle::dialect::FetchOp>(divide, "out1", 0);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));
  ASSERT_EQ(program.block()->size(), 2u);
}

TEST(IROpFusionPass, layer_norm2) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());
  auto inputs = BuildInput(&builder, {{128, 128, 768}, {768}, {768}});

  std::vector<int64_t> axes{-1};
  auto num = builder
                 .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
                                                 768.0,
                                                 phi::DataType::FLOAT32,
                                                 phi::CPUPlace())
                 .result(0);
  auto eps = builder
                 .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
                                                 1e-5,
                                                 phi::DataType::FLOAT32,
                                                 phi::CPUPlace())
                 .result(0);
  auto sum = builder.Build<cinn::dialect::ReduceSumOp>(inputs[0], axes, true)
                 .result(0);
  std::vector<int64_t> all_axes{0, 1, 2};
  std::vector<int64_t> out_shape1{128, 128, 1};
  auto num1 =
      builder.Build<cinn::dialect::BroadcastOp>(num, all_axes, out_shape1)
          .result(0);
  auto mean = builder.Build<paddle::dialect::DivideOp>(sum, num1).result(0);
  auto power = builder.Build<paddle::dialect::MultiplyOp>(inputs[0], inputs[0])
                   .result(0);
  auto power_sum =
      builder.Build<cinn::dialect::ReduceSumOp>(power, axes, true).result(0);
  auto mean2 =
      builder.Build<paddle::dialect::DivideOp>(power_sum, num1).result(0);
  auto power_mean =
      builder.Build<paddle::dialect::MultiplyOp>(mean, mean).result(0);
  auto var =
      builder.Build<paddle::dialect::SubtractOp>(mean2, power_mean).result(0);
  std::vector<int64_t> out_shape2{128, 128, 768};
  auto sub =
      builder.Build<paddle::dialect::SubtractOp>(inputs[0], mean).result(0);
  auto eps1 =
      builder.Build<cinn::dialect::BroadcastOp>(eps, all_axes, out_shape2)
          .result(0);
  auto t1 = builder.Build<paddle::dialect::AddOp>(var, eps1).result(0);
  auto t2 = builder.Build<paddle::dialect::SqrtOp>(t1).result(0);
  auto t3 = builder.Build<paddle::dialect::DivideOp>(sub, t2).result(0);
  auto scale =
      builder.Build<cinn::dialect::BroadcastOp>(inputs[1], all_axes, out_shape2)
          .result(0);
  auto bias =
      builder.Build<cinn::dialect::BroadcastOp>(inputs[2], all_axes, out_shape2)
          .result(0);
  auto t5 = builder.Build<paddle::dialect::MultiplyOp>(t3, scale).result(0);

  auto out1 = builder.Build<paddle::dialect::MultiplyOp>(t5, bias).result(0);
  auto out2 =
      builder.Build<cinn::dialect::ReshapeOp>(mean, std::vector<int>({-1}))
          .result(0);
  auto out3 =
      builder.Build<cinn::dialect::ReshapeOp>(mean2, std::vector<int>({-1}))
          .result(0);

  builder.Build<paddle::dialect::FetchOp>(out1, "out1", 0);
  builder.Build<paddle::dialect::FetchOp>(out2, "out2", 1);
  builder.Build<paddle::dialect::FetchOp>(out3, "out3", 2);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());

  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Fatal("Pass manager run failed."));

  // TODO(phlrain): need update same as 4u
  ASSERT_EQ(program.block()->size(), 10u);
}
