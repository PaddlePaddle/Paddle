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
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"

std::vector<pir::OpResult> BuildInput(
    ::pir::Builder* builder,
    const std::vector<std::vector<int64_t>>& vec_shapes) {
  std::vector<pir::OpResult> vec_res;
  for (size_t i = 0; i < vec_shapes.size(); ++i) {
    auto op = builder->Build<paddle::dialect::FullOp>(
        vec_shapes[i], 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

    vec_res.push_back(op.result(0));
  }

  return vec_res;
}

TEST(MapExpr, ElementWise_Fusion_0) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ::pir::Program program_base(ctx);
  ::pir::Builder builder_base = ::pir::Builder(ctx, program_base.block());

  int h = 32, w = 32;
  auto inputs = BuildInput(&builder_base, {{h, w}, {h, w}, {h, w}, {h, w}});

  ::pir::Program program(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program.block());

  auto e =
      builder.Build<paddle::dialect::AddOp>(inputs[0], inputs[1]).result(0);
  auto f = builder.Build<paddle::dialect::AddOp>(e, inputs[2]).result(0);
  builder.Build<paddle::dialect::AddOp>(f, inputs[2]);

  std::vector<pir::Operation*> vec_op;
  for (auto& op : *program.block()) {
    vec_op.push_back(&op);
  }

  auto res = cinn::dialect::ir::OpFusionPassInternal(vec_op);

  auto new_group = cinn::dialect::ir::GeneralFusionMergePassInternal(res);

  ASSERT_EQ(res.size(), 1u);

  ASSERT_EQ(res[0]->ops.size(), program.block()->size());
}
