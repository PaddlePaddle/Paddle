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
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/divide_group_op_to_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/lower_cinn_fusion_op_pass.h"
#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass_manager.h"

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
      CreateDenseTensorTypes(common::make_ddim(shape)));
  pir::Block* block1 = group_op1.block();
  builder.SetInsertionPointToBlockEnd(block1);
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{full_op_x.out()});

  builder.SetInsertionPointToBlockEnd(program->block());
  auto group_op2 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim(shape)));
  pir::Block* block2 = group_op2.block();
  builder.SetInsertionPointToBlockEnd(block2);

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
  for (auto& sub_op : *(program->block())) {
    EXPECT_TRUE(sub_op.isa<cinn::dialect::GroupOp>());
    EXPECT_EQ(sub_op.dyn_cast<cinn::dialect::GroupOp>().GetOperators().size(),
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
  builder.SetInsertionPointToBlockEnd(block1.get());
  auto full_op_x = builder.Build<paddle::dialect::FullOp>(
      shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace());
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{full_op_x.out()});

  builder.SetInsertionPointToBlockEnd(program->block());
  auto group_op1 = builder.Build<cinn::dialect::GroupOp>(std::move(block1));

  // ------- Group op 2 ---------
  std::unique_ptr<::pir::Block> block2(new ::pir::Block());
  builder.SetInsertionPointToBlockEnd(block2.get());
  auto tan_op_x = builder.Build<paddle::dialect::TanOp>(group_op1->result(0));
  auto relu_op_x = builder.Build<paddle::dialect::ReluOp>(tan_op_x->result(0));
  auto tan_op_y = builder.Build<paddle::dialect::TanOp>(relu_op_x->result(0));
  auto relu_op_y = builder.Build<paddle::dialect::ReluOp>(tan_op_y->result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{relu_op_y.out()});

  builder.SetInsertionPointToBlockEnd(program->block());
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
  for (auto& sub_op : *(program->block())) {
    EXPECT_TRUE(sub_op.isa<cinn::dialect::GroupOp>());
    EXPECT_EQ(sub_op.dyn_cast<cinn::dialect::GroupOp>().GetOperators().size(),
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
      CreateDenseTensorTypes(common::make_ddim(shape)));
  pir::Block* block1 = group_op1.block();
  builder.SetInsertionPointToBlockEnd(block1);
  auto sin = builder.Build<paddle::dialect::SinOp>(full_x->result(0));

  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{
      sin.out(),
  });

  builder.SetInsertionPointToBlockEnd(program->block());
  auto group_op2 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim(shape)));
  pir::Block* block2 = group_op2.block();
  builder.SetInsertionPointToBlockEnd(block2);
  auto cos_op = builder.Build<paddle::dialect::CosOp>(full_y->result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{cos_op.out()});

  builder.SetInsertionPointToBlockEnd(program->block());
  auto group_op3 = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim(shape)));
  pir::Block* block3 = group_op3.block();
  builder.SetInsertionPointToBlockEnd(block3);
  auto add = builder.Build<paddle::dialect::AddOp>(group_op1->result(0),
                                                   group_op2->result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{add.out()});

  builder.SetInsertionPointToBlockEnd(program->block());
  auto exp = builder.Build<paddle::dialect::ExpOp>(group_op3->result(0));

  builder.Build<paddle::dialect::FetchOp>(exp.out(), "out", 0);
  return program;
}

TEST(GroupOp, CINNLowering) {
  // Step 1: Construct pir::Program
  std::shared_ptr<::pir::Program> program = BuildGroupProgramForLowering();

  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::PassManager pass_manager(ctx);
  pass_manager.AddPass(cinn::dialect::ir::CreateDivideGroupOpToFusionOpPass());
  pass_manager.AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());
  pass_manager.Run(program.get());

  paddle::platform::Place place = paddle::platform::CUDAPlace(0);

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(program.get(), place);

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

class GroupOpPattern : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::GroupOp>::OpRewritePattern;
  using Group = cinn::hlir::framework::pir::Group;

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto* program = group_op->GetParentProgram();
    ::pir::Builder builder = ::pir::Builder(ctx, program->block());
    VLOG(4) << "Before GroupOpPattern: " << *program;
    std::vector<::pir::Operation*> group_ops = group_op.GetOperators();
    auto yield_op = group_ops.back();
    std::vector<::pir::Type> output_type{yield_op->operand_source(0).type()};

    // construct hlir::Group
    Group group({group_ops.begin(), group_ops.end() - 1});
    group.input_ops[group_ops[0]] = 0;  // first tan
    auto last_op_idx = group_ops.size() - 2;
    group.output_ops.insert(group_ops[last_op_idx]);  // last relu

    // clone group and sync their op into new GroupOp
    builder.SetInsertionPointAfter(group_op.operation());
    auto new_group_op = builder.Build<cinn::dialect::GroupOp>(output_type);

    // prepare IrMapping
    ::pir::IrMapping ir_mapping;
    auto depend_value = group_ops[0]->operand_source(0);
    ir_mapping.Add(depend_value, depend_value);
    std::shared_ptr<Group> new_group =
        group.Clone(new_group_op.block(), ir_mapping);

    EXPECT_EQ(new_group->ops.size(), group.ops.size());
    EXPECT_EQ(new_group->input_ops.size(), group.input_ops.size());
    EXPECT_EQ(new_group->output_ops.size(), group.output_ops.size());

    // Add yield op
    builder.SetInsertionPointToBlockEnd(new_group_op.block());
    std::vector<::pir::Value> yield_inputs{
        new_group_op.GetOperators().back()->result(0)};
    builder.Build<::pir::YieldOp>(yield_inputs);
    EXPECT_EQ(new_group_op.GetOperators().size(), group_ops.size());

    // replace result UD between GroupOp
    rewriter.ReplaceAllUsesWith(group_op->result(0), new_group_op->result(0));
    rewriter.EraseOp(group_op);
    VLOG(4) << "After GroupOpPattern.EraseOp: " << *program;
    return true;
  }
};

class TestGroupClonePass : public pir::PatternRewritePass {
 public:
  TestGroupClonePass() : pir::PatternRewritePass("test_group_clone", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<GroupOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

std::shared_ptr<::pir::Program> BuildSingleGroupProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());
  const std::vector<int64_t> shape = {64, 128};
  // full op
  auto full_x = builder.Build<paddle::dialect::FullOp>(
      shape, 0.5, phi::DataType::FLOAT32, phi::GPUPlace());

  // group op
  auto group_op = builder.Build<cinn::dialect::GroupOp>(
      CreateDenseTensorTypes(common::make_ddim(shape)));
  pir::Block* block = group_op.block();
  builder.SetInsertionPointToBlockEnd(block);

  auto tan_op_x = builder.Build<paddle::dialect::TanOp>(full_x->result(0));
  auto relu_op_x = builder.Build<paddle::dialect::ReluOp>(tan_op_x->result(0));
  auto tan_op_y = builder.Build<paddle::dialect::TanOp>(relu_op_x->result(0));
  auto relu_op_y = builder.Build<paddle::dialect::ReluOp>(tan_op_y->result(0));
  builder.Build<::pir::YieldOp>(std::vector<::pir::Value>{relu_op_y.out()});

  // tan op
  builder.SetInsertionPointToBlockEnd(program->block());
  auto final_op = builder.Build<paddle::dialect::TanOp>(group_op->result(0));

  return program;
}

TEST(Group, Clone) {
  // Step 1: Construct pir::Program
  std::shared_ptr<::pir::Program> program = BuildSingleGroupProgram();
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ::pir::PassManager pm(ctx);
  // Step 2: Run TestGroupClonePass
  pm.AddPass(std::make_unique<TestGroupClonePass>());
  pm.Run(program.get());
}
