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
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_lowering_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass.h"
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

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  // full -> softmax(max -> subtract -> exp -> sum -> divide)
  const float value_one = 1.0;
  const std::vector<int64_t> shape = {128, 128, 768};
  auto x = builder
               .Build<paddle::dialect::FullOp>(
                   shape, value_one, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);

  auto max =
      builder.Build<paddle::dialect::MaxOp>(x, std::vector<int64_t>{-1}, true)
          .result(0);
  auto sub = builder.Build<paddle::dialect::SubtractOp>(x, max).result(0);
  auto exp = builder.Build<paddle::dialect::ExpOp>(sub).result(0);
  auto sum =
      builder
          .Build<paddle::dialect::SumOp>(
              exp, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
          .result(0);
  auto out = builder.Build<paddle::dialect::DivideOp>(exp, sum).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

TEST(GroupOp, TestBuild) {
  // Step 1: Construct pir::Program
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  std::shared_ptr<::pir::Program> program = BuildGroupProgram();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  cinn::dialect::ir::PdOp2CinnOpConverter(program.get());

  pir::PassManager pm(ctx);
  pm.AddPass(
      std::make_unique<cinn::dialect::ir::AddBroadcastToElementwisePass>());
  pm.AddPass(pir::CreateBuildCinnPass());
  CHECK_EQ(pm.Run(program.get()), true);

  auto res = cinn::dialect::ir::CINNGroupLoweringPass(program.get());

  paddle::platform::Place place = paddle::platform::CUDAPlace(0);

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(res.get(), place);

  paddle::framework::Scope exe_scope;

  paddle::framework::InterpreterCore executor(
      place, {"out@fetch"}, kernel_program->block(), &exe_scope);

  executor.Run({}, true);

  auto out_tensor =
      executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>();

  bool res0 = simple_cmp(out_tensor.data<float>()[0], 1.0 / 768);
  EXPECT_EQ(res0, true);
}

std::shared_ptr<::pir::Program> BuildLayerNormProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  std::vector<int64_t> axes{-1};
  auto x =
      builder
          .Build<paddle::dialect::FullOp>(std::vector<int64_t>({128, 128, 768}),
                                          1.0,
                                          phi::DataType::FLOAT32,
                                          phi::GPUPlace())
          .result(0);

  auto bias = builder
                  .Build<paddle::dialect::FullOp>(std::vector<int64_t>({768}),
                                                  1.0,
                                                  phi::DataType::FLOAT32,
                                                  phi::GPUPlace())
                  .result(0);

  auto scale = builder
                   .Build<paddle::dialect::FullOp>(std::vector<int64_t>({768}),
                                                   1.0,
                                                   phi::DataType::FLOAT32,
                                                   phi::GPUPlace())
                   .result(0);

  auto num = builder
                 .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
                                                 768.0,
                                                 phi::DataType::FLOAT32,
                                                 phi::CPUPlace())
                 .result(0);
  auto eps = builder
                 .Build<paddle::dialect::FullOp>(std::vector<int64_t>{1},
                                                 1e-5,
                                                 phi::DataType::FLOAT32,
                                                 phi::CPUPlace())
                 .result(0);

  auto sum =
      builder
          .Build<paddle::dialect::SumOp>(x, axes, phi::DataType::FLOAT32, true)
          .result(0);

  auto mean = builder.Build<paddle::dialect::DivideOp>(sum, num).result(0);
  auto power = builder.Build<paddle::dialect::MultiplyOp>(x, x).result(0);
  auto power_sum = builder
                       .Build<paddle::dialect::SumOp>(
                           power, axes, phi::DataType::FLOAT32, true)
                       .result(0);
  auto mean2 =
      builder.Build<paddle::dialect::DivideOp>(power_sum, num).result(0);
  auto power_mean =
      builder.Build<paddle::dialect::MultiplyOp>(mean, mean).result(0);

  auto var =
      builder.Build<paddle::dialect::SubtractOp>(mean2, power_mean).result(0);

  auto sub = builder.Build<paddle::dialect::SubtractOp>(x, mean).result(0);
  auto t1 = builder.Build<paddle::dialect::AddOp>(var, eps).result(0);
  auto t2 = builder.Build<paddle::dialect::SqrtOp>(t1).result(0);
  auto t3 = builder.Build<paddle::dialect::DivideOp>(sub, t2).result(0);
  auto t5 = builder.Build<paddle::dialect::MultiplyOp>(t3, scale).result(0);
  auto out = builder.Build<paddle::dialect::MultiplyOp>(t5, bias).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

TEST(GroupOp, TestBuildLayerNorm) {
  // Step 1: Construct pir::Program
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  std::shared_ptr<::pir::Program> program = BuildLayerNormProgram();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  cinn::dialect::ir::PdOp2CinnOpConverter(program.get());

  pir::PassManager pm(ctx);
  pm.AddPass(
      std::make_unique<cinn::dialect::ir::AddBroadcastToElementwisePass>());
  pm.AddPass(pir::CreateBuildCinnPass());
  CHECK_EQ(pm.Run(program.get()), true);

  auto res = cinn::dialect::ir::CINNGroupLoweringPass(program.get());

  paddle::platform::Place place = paddle::platform::CUDAPlace(0);

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(res.get(), place);

  paddle::framework::Scope exe_scope;

  paddle::framework::InterpreterCore executor(
      place, {"out@fetch"}, kernel_program->block(), &exe_scope);

  executor.Run({}, true);
  auto out_tensor =
      executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>();
}
