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
#include <sstream>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

std::vector<::pir::Type> CreateDenseTensorTypes(const phi::DDim& dims) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ::pir::Type fp32_dtype = ::pir::Float32Type::get(ctx);
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LoD lod = {};
  size_t offset = 0;
  std::vector<::pir::Type> op_output_types = {::pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset)};
  return op_output_types;
}

std::shared_ptr<::pir::Program> BuildGroupProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const float value_one = 1.0;
  const std::vector<int64_t> shape = {64, 128};
  auto group_op1 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(phi::make_ddim(shape)));
  pir::Block* block1 = group_op1.Block();
  builder.SetInsertionPointToEnd(block1);
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{full_op_x.out()});

  builder.SetInsertionPointToEnd(program->block());
  auto group_op2 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(phi::make_ddim(shape)));
  pir::Block* block2 = group_op2.Block();
  builder.SetInsertionPointToEnd(block2);

  auto tan_op_x = builder.Build<paddle::dialect::TanOp>(group_op1->result(0));
  auto relu_op_x = builder.Build<paddle::dialect::ReluOp>(tan_op_x->result(0));
  auto tan_op_y = builder.Build<paddle::dialect::TanOp>(relu_op_x->result(0));
  auto relu_op_y = builder.Build<paddle::dialect::ReluOp>(tan_op_y->result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{relu_op_y.out()});
  return program;
}

TEST(GroupOp, TestBuild) {
  // Step 1: Construct pir::Program
  std::shared_ptr<::pir::Program> program = BuildGroupProgram();
  std::stringstream ss;
  program->Print(ss);
  LOG(INFO) << ss.str();

  EXPECT_EQ(program->block()->size(), 2u);
  LOG(INFO) << program->block()->size();
  std::vector<uint32_t> op_num = {2, 5};
  int i = 0;
  for (auto* sub_op : *(program->block())) {
    EXPECT_TRUE(sub_op->isa<cinn::dialect::GroupOp>());
    EXPECT_EQ(sub_op->dyn_cast<cinn::dialect::GroupOp>().Ops().size(),
              op_num[i]);
    ++i;
  }
}
