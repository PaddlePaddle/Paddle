
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
#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_store_in_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/divide_group_op_to_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lower_cinn_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/merge_reshape_with_broadcast_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

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

static void RunAndCheckResult(::pir::Program* program,
                              const bool check_result = true,
                              const float gt_val = 2.0) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());
  pm.AddPass(cinn::dialect::ir::CreateAddBroadcastToElementwisePass());
  pm.AddPass(
      std::make_unique<cinn::dialect::ir::MergeReshapeWithBroadcastPass>());

  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());
  pm.AddPass(cinn::dialect::ir::CreateAddStoreInFusionOpPass());
  pm.AddPass(pir::CreateDeadCodeEliminationPass());
  pm.AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());
  pm.EnableIRPrinting();
  CHECK_EQ(pm.Run(program), true);

  paddle::platform::Place place = paddle::platform::CUDAPlace(0);

  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(program, place);

  paddle::framework::Scope exe_scope;

  paddle::framework::InterpreterCore executor(
      place, {"out@fetch"}, kernel_program->block(), &exe_scope);

  executor.Run({}, true);

  auto out_tensor =
      executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>();

  if (check_result) {
    std::cerr << "res  " << out_tensor.data<float>()[0] << std::endl;
    bool res0 = simple_cmp(out_tensor.data<float>()[0], gt_val);
    EXPECT_EQ(res0, true);
  }
}

// std::shared_ptr<::pir::Program> BuildReshapeSumProgram() {
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
//   auto program = std::make_shared<::pir::Program>(ctx);
//   ::pir::Builder builder = ::pir::Builder(ctx, program->block());

//   const std::vector<int64_t> x_shape = {-1, -1};
//   const std::vector<int64_t> y_shape = {-1, 1};

//   auto x = builder
//                 .Build<paddle::dialect::DataOp>(
//                     "input_x", x_shape, phi::DataType::FLOAT32,
//                     phi::GPUPlace())
//                 .result(0);

//   auto y = builder
//                .Build<paddle::dialect::DataOp>(
//                    "input_y", y_shape, phi::DataType::FLOAT32,
//                    phi::GPUPlace())
//                .result(0);
//   auto t = builder.Build< pir::CombineOp>( std::vector<pir::Value>({x, y})
//   ).result(0); auto concat = builder.Build<paddle::dialect::ConcatOp>( t,
//   1).result(0); auto sum = builder
//                  .Build<paddle::dialect::SinOp>(
//                      concat)
//                  .result(0);

//   builder.Build<paddle::dialect::FetchOp>(sum, "out", 0);
//   return program;
// }

std::shared_ptr<::pir::Program> BuildReshapeSumProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const std::vector<int64_t> x_shape = {-1, -1};
  const std::vector<int64_t> y_shape = {-1, -1};

  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "input_x", x_shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);

  auto y = builder
               .Build<paddle::dialect::DataOp>(
                   "input_y", y_shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);

  auto t = builder.Build<paddle::dialect::ShapeOp>(y).result(0);
  auto s1 = builder
                .Build<paddle::dialect::SliceOp>(t,
                                                 std::vector<int64_t>({0}),
                                                 std::vector<int64_t>({0}),
                                                 std::vector<int64_t>({1}),
                                                 std::vector<int64_t>({}),
                                                 std::vector<int64_t>({}))
                .result(0);

  auto s2 = builder
                .Build<paddle::dialect::FullOp>(std::vector<int64_t>({1}),
                                                -1,
                                                phi::DataType::INT64,
                                                phi::CPUPlace())
                .result(0);
  auto combine =
      builder.Build<pir::CombineOp>(std::vector<pir::Value>({s1, s2}))
          .result(0);
  //  auto s2 = builder.Build<paddle::dialect::AddOp>( s1, s1 ).result(0);
  auto out = builder.Build<paddle::dialect::ReshapeOp>(x, combine).result(0);

  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

TEST(GroupOp, TestBuildReshapeSum) {
  // Step 1: Construct pir::Program
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  std::shared_ptr<::pir::Program> program = BuildReshapeSumProgram();

  cinn::dialect::ir::ApplyCinnPass(program.get(), [&] {
    pir::IrContext* ctx = pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
    auto pass_manager =
        std::make_shared<::pir::PassManager>(::pir::IrContext::Instance(), 2);
    pass_manager->EnableIRPrinting();
    return pass_manager;
  });

  // RunAndCheckResult(program.get(), true, 128 * 128);
}
