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
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/general/constant_folding_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/gpu/multihead_matmul_fuse_pass.h"

#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass_manager.h"

#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(multihead_matmul, GPU, ALL_LAYOUT);

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp matmul_1_in_1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 300, 256},
                                             0.9,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  // The first path to matmul with scale (q).
  paddle::dialect::FullOp matmul_1_in_2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{256, 256},
                                             1.1,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::MatmulOp matmul_1 = builder.Build<paddle::dialect::MatmulOp>(
      matmul_1_in_1.out(), matmul_1_in_2.out(), false, false);

  paddle::dialect::FullOp add_1_in_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{256}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::AddOp add_1 =
      builder.Build<paddle::dialect::AddOp>(matmul_1.out(), add_1_in_2.out());

  paddle::dialect::ReshapeOp reshape_1 =
      builder.Build<paddle::dialect::ReshapeOp>(
          add_1.out(), std::vector<int64_t>{0, 0, 8, 32});

  paddle::dialect::TransposeOp transpose_1 =
      builder.Build<paddle::dialect::TransposeOp>(reshape_1.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  paddle::dialect::ScaleOp scale_op = builder.Build<paddle::dialect::ScaleOp>(
      transpose_1.out(), 0.1767766922712326, 0.0, true);

  // The second path to matmul (k).
  paddle::dialect::FullOp matmul_2_in_2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{256, 256},
                                             1.1,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  paddle::dialect::MatmulOp matmul_2 = builder.Build<paddle::dialect::MatmulOp>(
      matmul_1_in_1.out(), matmul_2_in_2.out(), false, false);

  paddle::dialect::FullOp add_2_in_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{256}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());
  paddle::dialect::AddOp add_op2 =
      builder.Build<paddle::dialect::AddOp>(matmul_2.out(), add_2_in_2.out());

  paddle::dialect::ReshapeOp reshape_2 =
      builder.Build<paddle::dialect::ReshapeOp>(
          add_op2.out(), std::vector<int64_t>{0, 0, 8, 32});

  paddle::dialect::TransposeOp transpose_2 =
      builder.Build<paddle::dialect::TransposeOp>(reshape_2.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  // The third path to matmul (v).
  paddle::dialect::FullOp matmul_3_in_2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{256, 256},
                                             1.1,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  paddle::dialect::MatmulOp matmul_3 = builder.Build<paddle::dialect::MatmulOp>(
      matmul_1_in_1.out(), matmul_3_in_2.out(), false, false);

  paddle::dialect::FullOp add_3_in_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{256}, 1.5, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::AddOp add_3 =
      builder.Build<paddle::dialect::AddOp>(matmul_3.out(), add_3_in_2.out());

  paddle::dialect::ReshapeOp reshape_3 =
      builder.Build<paddle::dialect::ReshapeOp>(
          add_3.out(), std::vector<int64_t>{0, 0, 8, 32});

  paddle::dialect::TransposeOp transpose_3 =
      builder.Build<paddle::dialect::TransposeOp>(reshape_3.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  // softmax(qk)v
  paddle::dialect::MatmulOp matmul_4 = builder.Build<paddle::dialect::MatmulOp>(
      scale_op.out(), transpose_2.out(), false, true);

  paddle::dialect::FullOp add_4_in_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1, 8, 300, 300},
      1.5,
      phi::DataType::FLOAT32,
      phi::CPUPlace());

  paddle::dialect::AddOp add_4 =
      builder.Build<paddle::dialect::AddOp>(matmul_4.out(), add_4_in_2.out());

  paddle::dialect::SoftmaxOp softmax_op =
      builder.Build<paddle::dialect::SoftmaxOp>(add_4.out(), -1);
  paddle::dialect::MatmulOp matmul_5 = builder.Build<paddle::dialect::MatmulOp>(
      softmax_op.out(), transpose_3.out(), false, false);

  paddle::dialect::TransposeOp transpose_4 =
      builder.Build<paddle::dialect::TransposeOp>(matmul_5.out(),
                                                  std::vector<int>{0, 2, 1, 3});

  paddle::dialect::ReshapeOp reshape_4 =
      builder.Build<paddle::dialect::ReshapeOp>(
          transpose_4.out(), std::vector<int64_t>{0, 0, 256});

  builder.Build<paddle::dialect::FetchOp>(reshape_4.out(), "out", 0);
}

TEST(DrrTest, AttentionFuse) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);
  EXPECT_EQ(program.block()->size(), 33u);

  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateMultiHeadMatmulFusePass());
  std::unique_ptr<pir::Pass> constant_folding_pass =
      pir::CreateConstantFoldingPass();
  constant_folding_pass->Set(pir::Pass::kPlaceAttr,
                             new phi::Place{phi::GPUPlace{}});
  constant_folding_pass->Set(pir::Pass::kParamScopeAttr,
                             new paddle::framework::Scope{});
  pm.AddPass(std::move(constant_folding_pass));
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.EnableIRPrinting();

  PADDLE_ENFORCE_EQ(pm.Run(&program),
                    true,
                    common::errors::Unavailable("pm fail to run program"));
  EXPECT_EQ(program.block()->size(), 2u);
}
