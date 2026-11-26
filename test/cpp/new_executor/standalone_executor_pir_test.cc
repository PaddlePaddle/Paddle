// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/standalone_executor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <string>

#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_interpreter.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"

#include "paddle/common/macros.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

DECLARE_FILE_SYMBOLS(kernel_dialect);

PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full_int_array, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sqrt, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(less_than, CPU, ALL_LAYOUT);

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

namespace paddle {
namespace framework {

TEST(StandaloneExecutor, run) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program((ctx));

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Builder builder = pir::Builder(ctx, program.block());

  paddle::dialect::FullOp op1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::FullOp op2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  auto add_op =
      builder.Build<paddle::dialect::AddOp>(op1->result(0), op2->result(0));

  std::string out_name = "add_out";
  builder.Build<pir::ShadowOutputOp>(add_op->result(0), out_name);

  auto kernel_program = pir::PdOpLowerToKernelPass(&program);

  auto place = phi::CPUPlace();
  Scope scope;

  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  test_core.SetSkipGcVars({out_name});

  test_core.Run({});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(out_name)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(out_name)->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 2.0);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 2.0);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 2.0);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 2.0);

  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}

TEST(StandaloneExecutor, run_error) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program((ctx));

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Builder builder = pir::Builder(ctx, program.block());

  paddle::dialect::FullOp op1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT32, phi::CPUPlace());

  paddle::dialect::FullOp op2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 1.0, phi::DataType::FLOAT64, phi::CPUPlace());

  auto add_op =
      builder.Build<paddle::dialect::AddOp>(op1->result(0), op2->result(0));

  std::string out_name = "add_out";
  builder.Build<pir::ShadowOutputOp>(add_op->result(0), out_name);

  auto kernel_program = pir::PdOpLowerToKernelPass(&program);

  for (auto op : kernel_program->block()->ops()) {
    op->erase_attribute("origin_id");
  }

  auto place = phi::CPUPlace();
  Scope scope;

  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  test_core.SetSkipGcVars({out_name});

  try {
    test_core.Run({});
  } catch (std::exception& e) {
    bool is_catch =
        std::string(e.what()).find("InvalidArgumentError") != std::string::npos;
    EXPECT_EQ(is_catch, true);
  }
}

TEST(StandaloneExecutor, run_feed_tensor) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program(ctx);

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Builder builder = pir::Builder(ctx, program.block());

  pir::OpInfo feed_op_info =
      ctx->GetRegisteredOpInfo(paddle::dialect::FeedOp::name());

  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DDim dims = {1};
  phi::DataLayout data_layout = phi::DataLayout::NCHW;
  phi::LegacyLoD lod = {{0}};
  size_t offset = 0;
  pir::Type dense_tensor_dtype = paddle::dialect::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  pir::AttributeMap attr_map1;
  attr_map1.insert(std::pair<std::string, pir::Attribute>(
      "name", pir::StrAttribute::get(ctx, "x")));
  attr_map1.insert(std::pair<std::string, pir::Attribute>(
      "col", pir::Int32Attribute::get(ctx, 0)));
  pir::Operation* feed_op1 =
      pir::Operation::Create({}, attr_map1, {dense_tensor_dtype}, feed_op_info);
  program.block()->push_back(feed_op1);

  pir::AttributeMap attr_map2;
  attr_map2.insert(std::pair<std::string, pir::Attribute>(
      "name", pir::StrAttribute::get(ctx, "y")));
  attr_map2.insert(std::pair<std::string, pir::Attribute>(
      "col", pir::Int32Attribute::get(ctx, 0)));
  pir::Operation* feed_op2 =
      pir::Operation::Create({}, attr_map2, {dense_tensor_dtype}, feed_op_info);
  program.block()->push_back(feed_op2);

  auto add_op = builder.Build<paddle::dialect::AddOp>(feed_op1->result(0),
                                                      feed_op2->result(0));
  std::string out_name = "add_out";
  builder.Build<pir::ShadowOutputOp>(add_op->result(0), out_name);

  auto kernel_program = pir::PdOpLowerToKernelPass(&program);

  auto place = phi::CPUPlace();
  Scope scope;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  test_core.SetSkipGcVars({out_name});

  phi::DenseTensorMeta meta(
      phi::DataType::FLOAT32, dims, data_layout, lod, offset);
  phi::DeviceContext* dev_ctx =
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace());

  phi::DenseTensor tensor_x;
  tensor_x.set_meta(meta);
  dev_ctx->Alloc(&tensor_x, phi::DataType::FLOAT32);
  float* tensor_x_data = tensor_x.data<float>();
  *tensor_x_data = 1.0;

  phi::DenseTensor tensor_y;
  tensor_y.set_meta(meta);
  dev_ctx->Alloc(&tensor_y, phi::DataType::FLOAT32);
  float* tensor_y_data = tensor_y.data<float>();
  *tensor_y_data = 2.0;

  test_core.Run({"x", "y"}, {tensor_x, tensor_y});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(out_name)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(out_name)->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 3.0);
  EXPECT_EQ(res0, true);
}

TEST(StandaloneExecutor, run_inplace_sqrt) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Program program((ctx));
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  pir::Builder builder = pir::Builder(ctx, program.block());

  paddle::dialect::FullOp full = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2, 2}, 4.0, phi::DataType::FLOAT32, phi::CPUPlace());

  builder.Build<paddle::dialect::Sqrt_Op>(full->result(0));

  std::string out_name = "full_out";
  builder.Build<pir::ShadowOutputOp>(full->result(0), out_name);

  auto kernel_program = pir::PdOpLowerToKernelPass(&program);

  auto place = phi::CPUPlace();
  Scope scope;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  test_core.SetSkipGcVars({out_name});

  test_core.Run({});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(out_name)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(out_name)->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 2.0);
  bool res1 = simple_cmp(out_tensor.data<float>()[1], 2.0);
  bool res2 = simple_cmp(out_tensor.data<float>()[2], 2.0);
  bool res3 = simple_cmp(out_tensor.data<float>()[3], 2.0);

  EXPECT_EQ(scope.kids().size(), 1u);
  EXPECT_EQ(scope.kids().front()->Size(), 2u);
  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
  EXPECT_EQ(res2, true);
  EXPECT_EQ(res3, true);
}

TEST(StandaloneExecutor, if_op) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  auto full_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1}, true, phi::DataType::BOOL);

  auto if_op = builder.Build<paddle::dialect::IfOp>(
      full_op.out(), std::vector<pir::Type>{full_op.result(0).type()});

  auto& true_block = if_op.true_block();

  builder.SetInsertionPointToStart(&true_block);

  auto full_op_1 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{2}, true, phi::DataType::BOOL);
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{full_op_1.out()});

  auto& false_block = if_op.false_block();

  builder.SetInsertionPointToStart(&false_block);

  auto full_op_2 = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{3}, true, phi::DataType::BOOL);
  builder.Build<pir::YieldOp>(std::vector<pir::Value>{full_op_2.out()});

  std::string out_name = "if_out";
  builder.SetInsertionPointToBlockEnd(block);
  builder.Build<pir::ShadowOutputOp>(if_op->result(0), out_name);

  auto kernel_program = pir::PdOpLowerToKernelPass(&program);

  auto place = phi::CPUPlace();
  Scope scope;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  test_core.SetSkipGcVars({out_name});

  test_core.Run({});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(out_name)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(out_name)->Get<phi::DenseTensor>();

  bool res0 = out_tensor.data<bool>()[0] == true;
  bool res1 = out_tensor.data<bool>()[1] == true;

  EXPECT_EQ(res0, true);
  EXPECT_EQ(res1, true);
}

using namespace paddle::dialect;  // NOLINT
TEST(StandaloneExecutor, while_op) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::ControlFlowDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  auto i = builder
               .Build<paddle::dialect::FullOp>(
                   std::vector<int64_t>{1}, 1, phi::DataType::INT32)
               .out();

  auto ten = builder
                 .Build<paddle::dialect::FullOp>(
                     std::vector<int64_t>{1}, 10, phi::DataType::INT32)
                 .out();

  // compute condition value: i <= ten
  auto cond_value = builder.Build<LessEqualOp>(i, ten).out();

  auto while_op =
      builder.Build<WhileOp>(cond_value, std::vector<pir::Value>{i, ten});

  // { i = i + 1}
  pir::Block& body_block = while_op.body();
  auto body_i_argument = body_block.arg(0);
  auto body_ten_argument = body_block.arg(1);
  builder.SetInsertionPointToStart(&body_block);
  auto one =
      builder.Build<FullOp>(std::vector<int64_t>{1}, 1, phi::DataType::INT32)
          .out();
  auto new_i = builder.Build<AddOp>(body_i_argument, one).out();

  // compute new condition value: new_i <= new_ten
  auto new_cond_value =
      builder.Build<LessEqualOp>(new_i, body_ten_argument).out();

  builder.Build<pir::YieldOp>(
      std::vector<pir::Value>{new_cond_value, new_i, body_ten_argument});

  builder.SetInsertionPointAfter(while_op);

  std::string out_name = "while_out";
  builder.Build<pir::ShadowOutputOp>(while_op->result(0), out_name);

  auto kernel_program = PdOpLowerToKernelPass(&program);

  auto place = phi::CPUPlace();
  Scope scope;
  InterpreterCore test_core(place, {}, kernel_program->block(), &scope);

  test_core.SetSkipGcVars({out_name});

  test_core.Run({});

  auto out_tensor =
      test_core.local_scope() == nullptr
          ? scope.FindVar(out_name)->Get<phi::DenseTensor>()
          : test_core.local_scope()->FindVar(out_name)->Get<phi::DenseTensor>();

  bool res0 = out_tensor.data<int>()[0] == 11;

  EXPECT_EQ(res0, true);
}

namespace {

class StubInstruction : public InstructionBase {
 public:
  StubInstruction(std::string name, pir::Operation* op)
      : InstructionBase(0, phi::CPUPlace()), name_(std::move(name)), op_(op) {}

  void Run() override {}
  const std::string& Name() const override { return name_; }
  pir::Operation* Operation() const override { return op_; }

 private:
  std::string name_;
  pir::Operation* op_;
};

pir::Operation* BuildFullKernelOpWithDims(pir::IrContext* ctx,
                                          pir::Program* storage,
                                          const std::vector<int64_t>& shape) {
  auto alloc_type =
      paddle::dialect::AllocatedDenseTensorType::get(ctx,
                                                     phi::CPUPlace(),
                                                     pir::Float32Type::get(ctx),
                                                     common::make_ddim(shape),
                                                     phi::DataLayout::NCHW,
                                                     phi::LegacyLoD{},
                                                     0);
  pir::AttributeMap attrs;
  attrs["op_name"] =
      pir::StrAttribute::get(ctx, paddle::dialect::FullOp::name());
  attrs["kernel_name"] = pir::StrAttribute::get(ctx, "full");
  attrs["kernel_key"] = paddle::dialect::KernelAttribute::get(
      ctx,
      phi::KernelKey(phi::Backend::CPU,
                     phi::DataLayout::ALL_LAYOUT,
                     phi::DataType::FLOAT32));
  auto op_info = ctx->GetRegisteredOpInfo(paddle::dialect::PhiKernelOp::name());
  pir::Operation* op = pir::Operation::Create({}, attrs, {alloc_type}, op_info);
  storage->block()->push_back(op);
  return op;
}

}  // namespace

TEST(StandaloneExecutor, is_lightweight_metadata_sync_op_boundaries) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<KernelDialect>();

  // Always-lightweight ops
  StubInstruction full_int_array_ins(FullIntArrayOp::name(), nullptr);
  StubInstruction shape_ins(ShapeOp::name(), nullptr);
  StubInstruction shape64_ins(Shape64Op::name(), nullptr);
  StubInstruction other_ins("pd_op.add", nullptr);
  EXPECT_TRUE(IsLightweightMetadataSyncOp(&full_int_array_ins));
  EXPECT_TRUE(IsLightweightMetadataSyncOp(&shape_ins));
  EXPECT_TRUE(IsLightweightMetadataSyncOp(&shape64_ins));
  EXPECT_FALSE(IsLightweightMetadataSyncOp(&other_ins));

  // FullOp boundary cases: 0-D scalar, zero-size tensor, 0-size with dynamic
  // dim, exact 64 threshold, 65 (> 64) rejected, overflow-like huge shape
  // rejected, dynamic -1 dim rejected, and null op rejected.
  StubInstruction null_full_ins(FullOp::name(), nullptr);
  EXPECT_FALSE(IsLightweightMetadataSyncOp(&null_full_ins));

  pir::Program holder(ctx);
  auto check_full = [&](const std::vector<int64_t>& shape, bool expected) {
    pir::Operation* op = BuildFullKernelOpWithDims(ctx, &holder, shape);
    StubInstruction ins(FullOp::name(), op);
    EXPECT_EQ(IsLightweightMetadataSyncOp(&ins), expected);
  };

  check_full({}, true);                             // 0-D scalar (numel = 1)
  check_full({1}, true);                            // 1-element tensor
  check_full({0, 1000}, true);                      // 0-size tensor (numel = 0)
  check_full({0, -1}, false);                       // 0-size with dynamic dim
  check_full({8, 8}, true);                         // exact threshold (64)
  check_full({64}, true);                           // exact threshold (64)
  check_full({65}, false);                          // just above threshold (65)
  check_full({8, 9}, false);                        // 72 > 64
  check_full({-1}, false);                          // dynamic shape
  check_full({4, -1}, false);                       // partially dynamic shape
  check_full({1000000000LL, 1000000000LL}, false);  // overflow-safe rejection
}

}  // namespace framework
}  // namespace paddle
