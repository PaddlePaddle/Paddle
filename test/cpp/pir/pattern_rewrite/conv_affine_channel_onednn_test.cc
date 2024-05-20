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
#include <memory>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/transforms/onednn/conv_affine_channel_onednn_fuse_pass.h"
#endif
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

void BuildProgram(pir::Builder &builder) {  // NOLINT
  paddle::dialect::FullOp full_input_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 4, 64, 64},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  paddle::dialect::FullOp full_filter_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4, 4, 1, 1},
                                             1.5,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  paddle::dialect::Conv2dOp conv2d_op =
      builder.Build<paddle::dialect::Conv2dOp>(full_input_op.out(),
                                               full_filter_op.out());

  paddle::dialect::FullOp ac_bias_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4}, 1.0);
  paddle::dialect::FullOp ac_scale_op =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{4}, 1.0);

  paddle::dialect::AffineChannelOp affine_channel_op =
      builder.Build<paddle::dialect::AffineChannelOp>(
          conv2d_op.out(), ac_scale_op.out(), ac_bias_op.out());
  builder.Build<paddle::dialect::FetchOp>(affine_channel_op.out(), "out", 0);
}
TEST(DrrTest, ConvAffineChanne) {
  paddle::platform::Place place = paddle::platform::CPUPlace();
  paddle::framework::Scope scope;
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);
  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateAffineChannelPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();
  // Run pass
  CHECK_EQ(pm.Run(&program), true);
}
