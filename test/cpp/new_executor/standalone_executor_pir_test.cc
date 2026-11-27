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
#include "paddle/fluid/framework/new_executor/interpreter/op_schedule_analyzer.h"
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
  StubInstruction(std::string name,
                  pir::Operation* op,
                  OpFuncType func_type = OpFuncType::kCpuSync)
      : InstructionBase(0, phi::CPUPlace()), name_(std::move(name)), op_(op) {
    SetKernelType(func_type);
  }

  void Run() override {}
  const std::string& Name() const override { return name_; }
  pir::Operation* Operation() const override { return op_; }

 private:
  std::string name_;
  pir::Operation* op_;
};

pir::Type MakeAllocTensorType(pir::IrContext* ctx,
                              const std::vector<int64_t>& shape,
                              const phi::Place& place = phi::CPUPlace()) {
  return paddle::dialect::AllocatedDenseTensorType::get(
      ctx,
      place,
      pir::Float32Type::get(ctx),
      common::make_ddim(shape),
      phi::DataLayout::NCHW,
      phi::LegacyLoD{},
      0);
}

pir::Operation* BuildPhiKernelOp(
    pir::IrContext* ctx,
    pir::Program* storage,
    const std::string& logical_op_name,
    const std::vector<pir::Value>& inputs,
    const std::vector<std::vector<int64_t>>& output_shapes,
    phi::Backend backend = phi::Backend::CPU,
    const phi::Place& output_place = phi::CPUPlace(),
    const char* kernel_op_name = paddle::dialect::PhiKernelOp::name()) {
  std::vector<pir::Type> result_types;
  result_types.reserve(output_shapes.size());
  for (const auto& shape : output_shapes) {
    result_types.push_back(MakeAllocTensorType(ctx, shape, output_place));
  }
  pir::AttributeMap attrs;
  attrs["op_name"] = pir::StrAttribute::get(ctx, logical_op_name);
  attrs["kernel_name"] = pir::StrAttribute::get(ctx, logical_op_name);
  attrs["kernel_key"] = paddle::dialect::KernelAttribute::get(
      ctx,
      phi::KernelKey(
          backend, phi::DataLayout::ALL_LAYOUT, phi::DataType::FLOAT32));
  auto op_info = ctx->GetRegisteredOpInfo(kernel_op_name);
  pir::Operation* op =
      pir::Operation::Create(inputs, attrs, result_types, op_info);
  storage->block()->push_back(op);
  return op;
}

}  // namespace

TEST(StandaloneExecutor, op_schedule_info_three_tier_classification) {
  using interpreter::AnalyzeOpScheduleInfo;
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<KernelDialect>();

  const phi::Place cpu_place = phi::CPUPlace();
  pir::Program holder(ctx);

  // --- Nullptr & direct pir::Operation* overload coverage ---
  {
    auto null_ins_info = AnalyzeOpScheduleInfo(
        static_cast<const InstructionBase*>(nullptr), cpu_place);
    EXPECT_EQ(null_ins_info.func_type, OpFuncType::kGpuAsync);
    EXPECT_EQ(null_ins_info.host_cost, HostSyncCost::kNone);

    auto null_op_cpu_info =
        AnalyzeOpScheduleInfo(static_cast<pir::Operation*>(nullptr), cpu_place);
    EXPECT_EQ(null_op_cpu_info.func_type, OpFuncType::kCpuSync);
    EXPECT_EQ(null_op_cpu_info.host_cost, HostSyncCost::kCompute);

    auto null_op_gpu_info = AnalyzeOpScheduleInfo(
        static_cast<pir::Operation*>(nullptr), phi::GPUPlace(0));
    EXPECT_EQ(null_op_gpu_info.func_type, OpFuncType::kGpuAsync);
    EXPECT_EQ(null_op_gpu_info.host_cost, HostSyncCost::kNone);

    pir::Operation* small_full =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{4}});
    auto op_info = AnalyzeOpScheduleInfo(small_full, cpu_place);
    EXPECT_EQ(op_info.func_type, OpFuncType::kCpuSync);
    EXPECT_EQ(op_info.host_cost, HostSyncCost::kInlineMetadata);
    EXPECT_EQ(op_info.estimated_host_numel, 4);

    // Null op fallback on a Tier 1a blocking op name.
    StubInstruction null_blocking_ins(
        "pd_op.memcpy_d2h", nullptr, OpFuncType::kCpuSync);
    auto blocking_info = AnalyzeOpScheduleInfo(&null_blocking_ins, cpu_place);
    EXPECT_EQ(blocking_info.host_cost, HostSyncCost::kBlockingDeviceIO);
    EXPECT_FALSE(blocking_info.CanInlineInTraceRun());
  }

  // --- Tier 0: kGpuAsync -> kNone ---
  {
    StubInstruction async_ins("pd_op.relu", nullptr, OpFuncType::kGpuAsync);
    auto info = AnalyzeOpScheduleInfo(&async_ins, cpu_place);
    EXPECT_EQ(info.func_type, OpFuncType::kGpuAsync);
    EXPECT_EQ(info.host_cost, HostSyncCost::kNone);
    EXPECT_EQ(info.estimated_host_numel, 0);
    EXPECT_FALSE(info.IsSync());
    EXPECT_TRUE(info.CanInlineInTraceRun());
  }

  // --- Tier 1a: Blocking device/host I/O ops -> kBlockingDeviceIO ---
  for (const std::string& blocking_op : {"pd_op.memcpy_d2h",
                                         "pd_op.memcpy_d2h_multi_io",
                                         "pd_op.read_file",
                                         "pd_op.coalesce_tensor",
                                         "pd_op.print",
                                         "pd_op.assert"}) {
    pir::Operation* op = BuildPhiKernelOp(ctx, &holder, blocking_op, {}, {{1}});
    StubInstruction ins(blocking_op, op, OpFuncType::kCpuSync);
    auto info = AnalyzeOpScheduleInfo(&ins, cpu_place);
    EXPECT_EQ(info.host_cost, HostSyncCost::kBlockingDeviceIO) << blocking_op;
    EXPECT_FALSE(info.CanInlineInTraceRun()) << blocking_op;
  }

  // --- Tier 1b: Heavy/superlinear CPU ops even with tiny static shapes ---
  for (const std::string& heavy_op : {"pd_op.svd",
                                      "pd_op.matmul",
                                      "pd_op.sort",
                                      "pd_op.argsort",
                                      "pd_op.unique",
                                      "pd_op.nonzero"}) {
    pir::Operation* op = BuildPhiKernelOp(ctx, &holder, heavy_op, {}, {{2, 2}});
    StubInstruction ins(heavy_op, op, OpFuncType::kCpuSync);
    auto info = AnalyzeOpScheduleInfo(&ins, cpu_place);
    EXPECT_EQ(info.host_cost, HostSyncCost::kCompute) << heavy_op;
    EXPECT_FALSE(info.CanInlineInTraceRun()) << heavy_op;
  }

  // --- Tier 1c: LegacyKernelOp (non-PhiKernelOp) stays kCompute ---
  {
    pir::Operation* legacy_op =
        BuildPhiKernelOp(ctx,
                         &holder,
                         "pd_op.scale",
                         {},
                         {{4}},
                         phi::Backend::CPU,
                         phi::CPUPlace(),
                         paddle::dialect::LegacyKernelOp::name());
    StubInstruction ins("pd_op.scale", legacy_op, OpFuncType::kCpuSync);
    auto info = AnalyzeOpScheduleInfo(&ins, cpu_place);
    EXPECT_EQ(info.host_cost, HostSyncCost::kCompute);
    EXPECT_FALSE(info.CanInlineInTraceRun());
  }

  // --- Tier 2: Intrinsic O(1) metadata ops (including nullptr op fallback) ---
  for (const std::string& intrinsic_op : {std::string(FullIntArrayOp::name()),
                                          std::string(ShapeOp::name()),
                                          std::string(Shape64Op::name()),
                                          std::string(NumelOp::name())}) {
    StubInstruction null_ins(intrinsic_op, nullptr, OpFuncType::kCpuSync);
    auto null_info = AnalyzeOpScheduleInfo(&null_ins, cpu_place);
    EXPECT_EQ(null_info.host_cost, HostSyncCost::kInlineMetadata)
        << intrinsic_op;
    EXPECT_EQ(null_info.estimated_host_numel, 1);
    EXPECT_TRUE(null_info.CanInlineInTraceRun());

    // Even as kGpuSync (how ShapeOp/Shape64Op are typed on GPU), Tier 2 stays
    // kInlineMetadata.
    pir::Operation* op =
        BuildPhiKernelOp(ctx, &holder, intrinsic_op, {}, {{4}});
    StubInstruction gpu_sync_ins(intrinsic_op, op, OpFuncType::kGpuSync);
    auto gpu_info = AnalyzeOpScheduleInfo(&gpu_sync_ins, cpu_place);
    EXPECT_EQ(gpu_info.host_cost, HostSyncCost::kInlineMetadata)
        << intrinsic_op;
    EXPECT_TRUE(gpu_info.CanInlineInTraceRun());
  }

  // --- Tier 2: builtin.combine of CPU tensors -> kInlineMetadata ---
  {
    pir::Operation* c1 =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{1}});
    pir::Operation* c2 =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{1}});
    auto combine_info = ctx->GetRegisteredOpInfo(pir::CombineOp::name());
    std::vector<pir::Type> elem_types = {c1->result(0).type(),
                                         c2->result(0).type()};
    pir::Type vec_type = pir::VectorType::get(ctx, elem_types);
    pir::Operation* combine_op = pir::Operation::Create(
        {c1->result(0), c2->result(0)}, {}, {vec_type}, combine_info);
    holder.block()->push_back(combine_op);
    StubInstruction ins(
        pir::CombineOp::name(), combine_op, OpFuncType::kCpuSync);
    auto info = AnalyzeOpScheduleInfo(&ins, cpu_place);
    EXPECT_EQ(info.host_cost, HostSyncCost::kInlineMetadata);
    EXPECT_EQ(info.estimated_host_numel, 2);
    EXPECT_TRUE(info.CanInlineInTraceRun());
  }

  // --- Tier 3: Generic static I/O cost model boundary checks ---
  StubInstruction null_full_ins(FullOp::name(), nullptr, OpFuncType::kCpuSync);
  EXPECT_FALSE(
      AnalyzeOpScheduleInfo(&null_full_ins, cpu_place).CanInlineInTraceRun());

  auto check_single_output = [&](const std::string& op_name,
                                 const std::vector<int64_t>& shape,
                                 bool expected_inline,
                                 int64_t expected_numel) {
    pir::Operation* op = BuildPhiKernelOp(ctx, &holder, op_name, {}, {shape});
    StubInstruction ins(op_name, op, OpFuncType::kCpuSync);
    auto info = AnalyzeOpScheduleInfo(&ins, cpu_place);
    EXPECT_EQ(info.CanInlineInTraceRun(), expected_inline) << op_name;
    if (expected_inline) {
      EXPECT_EQ(info.host_cost, HostSyncCost::kInlineMetadata);
      EXPECT_EQ(info.estimated_host_numel, expected_numel);
    } else {
      EXPECT_EQ(info.host_cost, HostSyncCost::kCompute);
    }
  };

  check_single_output(FullOp::name(), {}, true, 1);         // 0-D scalar
  check_single_output(FullOp::name(), {1}, true, 1);        // 1-elem
  check_single_output(FullOp::name(), {0, 1000}, true, 1);  // 0-size clamped>=1
  check_single_output(FullOp::name(), {0, -1}, false, 0);   // 0-size + dynamic
  check_single_output(FullOp::name(), {8, 8}, true, 64);    // exact threshold
  check_single_output(FullOp::name(), {64}, true, 64);      // exact threshold
  check_single_output(FullOp::name(), {65}, false, 0);      // above threshold
  check_single_output(FullOp::name(), {8, 9}, false, 0);    // 72 > 64
  check_single_output(FullOp::name(), {-1}, false, 0);      // dynamic
  check_single_output(FullOp::name(), {4, -1}, false, 0);   // partial dynamic
  check_single_output(
      FullOp::name(), {1000000000LL, 1000000000LL}, false, 0);  // overflow-safe

  // Tier 3 with inputs + outputs (shape arithmetic like pd_op.scale /
  // pd_op.add)
  {
    pir::Operation* lhs =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{16}});
    pir::Operation* rhs =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{16}});
    // Total I/O numel = 16 + 16 + 16 = 48 <= 64 -> kInlineMetadata.
    pir::Operation* add_small = BuildPhiKernelOp(
        ctx, &holder, "pd_op.add", {lhs->result(0), rhs->result(0)}, {{16}});
    StubInstruction small_ins("pd_op.add", add_small, OpFuncType::kCpuSync);
    auto small_info = AnalyzeOpScheduleInfo(&small_ins, cpu_place);
    EXPECT_EQ(small_info.host_cost, HostSyncCost::kInlineMetadata);
    EXPECT_EQ(small_info.estimated_host_numel, 48);

    // Total I/O numel = 32 + 32 + 32 = 96 > 64 -> kCompute even though each
    // single tensor is <= 64.
    pir::Operation* big_lhs =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{32}});
    pir::Operation* big_rhs =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{32}});
    pir::Operation* add_over =
        BuildPhiKernelOp(ctx,
                         &holder,
                         "pd_op.add",
                         {big_lhs->result(0), big_rhs->result(0)},
                         {{32}});
    StubInstruction over_ins("pd_op.add", add_over, OpFuncType::kCpuSync);
    auto over_info = AnalyzeOpScheduleInfo(&over_ins, cpu_place);
    EXPECT_EQ(over_info.host_cost, HostSyncCost::kCompute);
    EXPECT_FALSE(over_info.CanInlineInTraceRun());
  }

  // Tier 3 rejects non-CPU input tensor (implicit cross-device transfer risk).
  {
    pir::Operation* gpu_src = BuildPhiKernelOp(ctx,
                                               &holder,
                                               FullOp::name(),
                                               {},
                                               {{4}},
                                               phi::Backend::GPU,
                                               phi::GPUPlace(0));
    pir::Operation* cpu_consumer = BuildPhiKernelOp(
        ctx, &holder, "pd_op.scale", {gpu_src->result(0)}, {{4}});
    StubInstruction ins("pd_op.scale", cpu_consumer, OpFuncType::kCpuSync);
    auto info = AnalyzeOpScheduleInfo(&ins, cpu_place);
    EXPECT_EQ(info.host_cost, HostSyncCost::kCompute);
    EXPECT_FALSE(info.CanInlineInTraceRun());
  }
}

TEST(StandaloneExecutor, count_trace_blocking_sync_ops_graph_budget) {
  using interpreter::AnalyzeOpScheduleInfo;
  using interpreter::CountTraceBlockingSyncOps;
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<KernelDialect>();

  const phi::Place cpu_place = phi::CPUPlace();
  pir::Program holder(ctx);

  // 16 ops * 64 elements = 1024 == kGraphInlineMetadataNumelBudget -> 0
  // blocking
  std::vector<std::unique_ptr<InstructionBase>> instrs;
  for (int i = 0; i < 16; ++i) {
    pir::Operation* op =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{64}});
    auto ins = std::make_unique<StubInstruction>(
        FullOp::name(), op, OpFuncType::kCpuSync);
    ins->SetScheduleInfo(AnalyzeOpScheduleInfo(ins.get(), cpu_place));
    instrs.push_back(std::move(ins));
  }
  EXPECT_EQ(CountTraceBlockingSyncOps(instrs), 0);

  // Adding 1 more lightweight op pushes cumulative numel to 1025 > 1024 ->
  // budget exceeded, counted as trace-blocking.
  {
    pir::Operation* extra_op =
        BuildPhiKernelOp(ctx, &holder, FullOp::name(), {}, {{1}});
    auto extra_ins = std::make_unique<StubInstruction>(
        FullOp::name(), extra_op, OpFuncType::kCpuSync);
    extra_ins->SetScheduleInfo(
        AnalyzeOpScheduleInfo(extra_ins.get(), cpu_place));
    instrs.push_back(std::move(extra_ins));
  }
  EXPECT_EQ(CountTraceBlockingSyncOps(instrs), 1);

  // Async ops never consume host budget or increment blocking count.
  {
    auto async_ins = std::make_unique<StubInstruction>(
        "pd_op.relu", nullptr, OpFuncType::kGpuAsync);
    async_ins->SetScheduleInfo(
        AnalyzeOpScheduleInfo(async_ins.get(), cpu_place));
    instrs.push_back(std::move(async_ins));
  }
  EXPECT_EQ(CountTraceBlockingSyncOps(instrs), 1);
}

}  // namespace framework
}  // namespace paddle
