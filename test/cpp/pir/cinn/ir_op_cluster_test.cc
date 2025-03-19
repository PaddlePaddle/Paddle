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
  PADDLE_ENFORCE_EQ(program.block()->size(),
                    2u,
                    common::errors::PreconditionNotMet(
                        "The number of FusionOp does not meet expectations。"));
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
  PADDLE_ENFORCE_EQ(program.block()->size(),
                    2u,
                    common::errors::PreconditionNotMet(
                        "The number of FusionOp does not meet expectations。"));
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
  PADDLE_ENFORCE_EQ(program.block()->size(),
                    3u,
                    common::errors::PreconditionNotMet(
                        "The number of FusionOp does not meet expectations。"));
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
  PADDLE_ENFORCE_EQ(program.block()->size(),
                    2u,
                    common::errors::PreconditionNotMet(
                        "The number of FusionOp does not meet expectations。"));
}
