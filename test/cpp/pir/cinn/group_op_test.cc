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
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_lowering_pass.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

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
  pir::Block* block1 = group_op1.block();
  builder.SetInsertionPointToEnd(block1);
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{full_op_x.out()});

  builder.SetInsertionPointToEnd(program->block());
  auto group_op2 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(phi::make_ddim(shape)));
  pir::Block* block2 = group_op2.block();
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
    EXPECT_EQ(sub_op->dyn_cast<cinn::dialect::GroupOp>().ops().size(),
              op_num[i]);
    ++i;
  }
}

std::shared_ptr<::pir::Program> BuildGroupProgramByBlock() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // ------- Group op 1 ---------
  const float value_one = 1.0;
  const std::vector<int64_t> shape = {64, 128};
  std::unique_ptr<::pir::Block> block1(new ::pir::Block());
  builder.SetInsertionPointToEnd(block1.get());
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{full_op_x.out()});

  builder.SetInsertionPointToEnd(program->block());
  auto group_op1 = builder.Build<cinn::dialect::GroupOp>(std::move(block1));

  // ------- Group op 2 ---------
  std::unique_ptr<::pir::Block> block2(new ::pir::Block());
  builder.SetInsertionPointToEnd(block2.get());
  auto tan_op_x = builder.Build<paddle::dialect::TanOp>(group_op1->result(0));
  auto relu_op_x = builder.Build<paddle::dialect::ReluOp>(tan_op_x->result(0));
  auto tan_op_y = builder.Build<paddle::dialect::TanOp>(relu_op_x->result(0));
  auto relu_op_y = builder.Build<paddle::dialect::ReluOp>(tan_op_y->result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{relu_op_y.out()});

  builder.SetInsertionPointToEnd(program->block());
  auto group_op2 = builder.Build<cinn::dialect::GroupOp>(std::move(block2));

  return program;
}

TEST(GroupOp, TestBuildByBlock) {
  // Step 1: Construct pir::Program
  std::shared_ptr<::pir::Program> program = BuildGroupProgramByBlock();
  std::stringstream ss;
  program->Print(ss);
  LOG(INFO) << ss.str();

  EXPECT_EQ(program->block()->size(), 2u);
  LOG(INFO) << program->block()->size();
  std::vector<uint32_t> op_num = {2, 5};
  int i = 0;
  for (auto* sub_op : *(program->block())) {
    EXPECT_TRUE(sub_op->isa<cinn::dialect::GroupOp>());
    EXPECT_EQ(sub_op->dyn_cast<cinn::dialect::GroupOp>().ops().size(),
              op_num[i]);
    ++i;
  }
}

std::shared_ptr<::pir::Program> BuildGroupProgramForLowering() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  const std::vector<int64_t> shape = {2, 2};
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());
  const float value = 0.5;
  auto full_x = builder.Build<paddle::dialect::FullOp>(
      shape, value, phi::DataType::FLOAT32, phi::GPUPlace());

  auto full_y = builder.Build<paddle::dialect::FullOp>(
      shape, value, phi::DataType::FLOAT32, phi::GPUPlace());

  auto group_op1 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(phi::make_ddim(shape)));
  pir::Block* block1 = group_op1.block();
  builder.SetInsertionPointToEnd(block1);
  auto sin = builder.Build<paddle::dialect::SinOp>(full_x->result(0));

  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{
      sin.out(),
  });

  builder.SetInsertionPointToEnd(program->block());
  auto group_op2 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(phi::make_ddim(shape)));
  pir::Block* block2 = group_op2.block();
  builder.SetInsertionPointToEnd(block2);
  auto cos_op = builder.Build<paddle::dialect::CosOp>(full_y->result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{cos_op.out()});

  builder.SetInsertionPointToEnd(program->block());
  auto group_op3 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(phi::make_ddim(shape)));
  pir::Block* block3 = group_op3.block();
  builder.SetInsertionPointToEnd(block3);
  auto add = builder.Build<paddle::dialect::AddOp>(group_op1->result(0),
                                                   group_op2->result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{add.out()});

  builder.SetInsertionPointToEnd(program->block());
  auto exp = builder.Build<paddle::dialect::ExpOp>(group_op3->result(0));

  builder.Build<paddle::dialect::FetchOp>(exp.out(), "out", 0);
  return program;
}

TEST(GroupOp, CINNLowering) {
  // Step 1: Construct pir::Program
  std::shared_ptr<::pir::Program> program = BuildGroupProgramForLowering();

  auto res = cinn::dialect::ir::CINNGroupLoweringPass(program.get());

  paddle::platform::Place place = paddle::platform::CUDAPlace(0);

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(res.get(), place);

  paddle::framework::Scope exe_scope;

  paddle::framework::InterpreterCore executor(
      place, {"out@fetch"}, kernel_program->block(), &exe_scope);

  std::set<std::string> out_names;
  out_names.insert("out@fetch");
  auto local_names = exe_scope.LocalVarNames();
  for (size_t i = 0; i < local_names.size(); ++i) {
    out_names.insert(local_names[i]);
  }

  executor.SetSkipGcVars(out_names);
  executor.Run({}, true);

  auto out_tensor =
      executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 3.88455);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 3.88455);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 3.88455);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 3.88455);

  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}
