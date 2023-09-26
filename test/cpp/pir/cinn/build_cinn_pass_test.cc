/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"

std::shared_ptr<::pir::Program> BuildAllOpSupportCinnGraph() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // full -> tan -> relu -> tan -> relu
  const float value_one = 1.0;
  const std::vector<int64_t> shape = {64, 128};
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());
  auto tan_op_x = builder.Build<paddle::dialect::TanOp>(full_op_x->result(0));
  auto relu_op_x = builder.Build<paddle::dialect::ReluOp>(tan_op_x->result(0));
  auto tan_op_y = builder.Build<paddle::dialect::TanOp>(relu_op_x->result(0));
  auto relu_op_y = builder.Build<paddle::dialect::ReluOp>(tan_op_y->result(0));

  return program;
}

TEST(BuildCinnPassTest, AllOpSupportCinn) {
  auto origin_program = BuildAllOpSupportCinnGraph();
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.EnablePassTiming();
  pm.EnableIRPrinting();
  CHECK_EQ(pm.Run(origin_program.get()), true);
  LOG(INFO) << "after pass: " << *origin_program;

  CHECK_EQ(origin_program->block()->size(), 1u);
  pir::Operation* group_op = origin_program->block()->front();
  pir::Block* group_block =
      group_op->dyn_cast<cinn::dialect::GroupOp>().Block();
  CHECK_EQ(group_block->size(), 6u);

  std::vector<std::string> op_names = {
      paddle::dialect::FullOp::name(),
      paddle::dialect::TanOp::name(),
      paddle::dialect::ReluOp::name(),
      paddle::dialect::TanOp::name(),
      paddle::dialect::ReluOp::name(),
      pir::YieldOp::name(),
  };
  int index = 0;
  for (auto iter : *group_block) {
    CHECK_EQ(iter->name(), op_names[index++]);
  }
}
