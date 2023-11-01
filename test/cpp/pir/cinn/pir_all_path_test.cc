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
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"
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
  const std::vector<int64_t> shape = {8, 8};
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

  program->Print(std::cout);

  cinn::dialect::ir::PdOp2CinnOpConverter(program.get());

  program->Print(std::cout);
  pir::PassManager pm(ctx);
  pm.AddPass(
      std::make_unique<cinn::dialect::ir::AddBroadcastToElementwisePass>());
  pm.AddPass(pir::CreateBuildCinnPass());
  CHECK_EQ(pm.Run(program.get()), true);
  std::cerr << "fin build cinn pass process " << std::endl;

  program->Print(std::cout);

  std::cerr << "finish here" << std::endl;

  auto res = cinn::dialect::ir::CINNGroupLoweringPass(program.get());

  res->Print(std::cout);

  paddle::platform::Place place = paddle::platform::CUDAPlace(0);

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(res.get(), place);

  kernel_program->Print(std::cout);

  paddle::framework::Scope exe_scope;

  paddle::framework::InterpreterCore executor(
      place, {"out@fetch"}, kernel_program->block(), &exe_scope);

  executor.Run({}, true);
  auto out_tensor =
      executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>();

  std::cerr << out_tensor << std::endl;
}

// TEST(GroupOp, TestBuildBadCAse) {
//   // Step 1: Construct pir::Program
//   ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//   std::shared_ptr<::pir::Program> program = BuildGroupProgram();

//   program->Print( std::cout );

//    pir::PassManager pm(ctx);
//   pm.AddPass(pir::CreateBuildCinnPass());
//   CHECK_EQ(pm.Run(program.get()), true);

//   program->Print( std::cout );

//   std::cerr << "finish here" << std::endl;
// }

// TEST(GroupOp, CINNLowering) {
//   // Step 1: Construct pir::Program
//   std::shared_ptr<::pir::Program> program = BuildGroupProgramForLowering();

//   auto res = cinn::dialect::ir::CINNGroupLoweringPass(program.get());

//   paddle::platform::Place place = paddle::platform::CUDAPlace(0);

//   auto kernel_program =
//       paddle::dialect::PdOpLowerToKernelPass(res.get(), place);

//   paddle::framework::Scope exe_scope;

//   paddle::framework::InterpreterCore executor(
//       place, {"out@fetch"}, kernel_program->block(), &exe_scope);

//   std::set<std::string> out_names;
//   out_names.insert("out@fetch");
//   auto local_names = exe_scope.LocalVarNames();
//   for (size_t i = 0; i < local_names.size(); ++i) {
//     out_names.insert(local_names[i]);
//   }

//   executor.SetSkipGcVars(out_names);
//   executor.Run({}, true);

//   auto out_tensor =
//       executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>();

//   bool res0 = simple_cmp(out_tensor.data<float>()[0], 3.88455);
//   bool res1 = simple_cmp(out_tensor.data<float>()[1], 3.88455);
//   bool res2 = simple_cmp(out_tensor.data<float>()[2], 3.88455);
//   bool res3 = simple_cmp(out_tensor.data<float>()[3], 3.88455);

//   EXPECT_EQ(res0, true);
//   EXPECT_EQ(res1, true);
//   EXPECT_EQ(res2, true);
//   EXPECT_EQ(res3, true);
// }
