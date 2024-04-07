// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/type.h"

TEST(layout_transformation_interface_test, operator) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  pir::Builder builder(ctx, program.block());

  auto build_input_value = [&](std::vector<int64_t> shape = {2, 2}) {
    auto uniform = builder.Build<paddle::dialect::UniformOp>(
        shape, phi::DataType::FLOAT32, 0.0, 1.0, 2, phi::CPUPlace());
    return uniform;
  };

  auto fused_conv = builder.Build<paddle::dialect::FusedConv2dAddActOp>(
      build_input_value(std::vector<int64_t>{2, 2, 2, 2}).result(0),
      build_input_value(std::vector<int64_t>{2, 2, 2, 2}).result(0),
      build_input_value().result(0),
      build_input_value().result(0));

  auto layout_transformation_iface =
      fused_conv->dyn_cast<paddle::dialect::LayoutTransformationInterface>();
  EXPECT_TRUE(layout_transformation_iface);

  EXPECT_EQ(layout_transformation_iface.PreferLayout(fused_conv),
            common::DataLayout::NHWC);
  EXPECT_NO_THROW(layout_transformation_iface.RewriteByLayout(
      fused_conv, common::DataLayout::NHWC));
  EXPECT_EQ(layout_transformation_iface.RelevantInputs(fused_conv).size(),
            fused_conv->operands().size());
  EXPECT_EQ(layout_transformation_iface.RelevantOutputs(fused_conv).size(),
            fused_conv->results().size());
}
