// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

COMMON_DECLARE_bool(print_ir);

bool simple_cmp(float a, float b) { return std::abs((a - b) / a) < 1e-5; }

template <typename T>
void FillValue(T* data, T value, size_t size) {
  for (int i = 0; i < size; ++i) {
    data[i] = value;
  }
}

std::shared_ptr<pir::PassManager> CreatePassManager() {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  auto pass_manager = std::make_shared<pir::PassManager>(ctx);
  if (FLAGS_print_ir) {
    pass_manager->EnableIRPrinting();
  }
  return pass_manager;
}

static void BuildAndRun(::pir::Program* program) {
  cinn::dialect::ir::ApplyCinnPass(program, CreatePassManager);

  paddle::platform::Place place = paddle::platform::CUDAPlace(0);
  auto kernel_program = paddle::dialect::PdOpLowerToKernelPass(program, place);

  paddle::framework::Scope exe_scope;
  paddle::framework::InterpreterCore executor(
      place, {"out@fetch"}, kernel_program->block(), &exe_scope);

  auto x_tensor = executor.local_scope()->FindVar("x")->Get<phi::DenseTensor>();
  LOG(INFO) << "x_tensor.rank: " << x_tensor.dims().size();
  phi::DDim ddim({128, 128, 768});
  size_t size = 128 * 128 * 768;
  x_tensor.ResizeAndAllocate(ddim);
  float* data = x_tensor.mutable_data<float>(ddim, place);
  LOG(INFO) << "x_tensor.rank: " << x_tensor.dims().size();

  executor.Run({"x"}, {x_tensor}, true);

  auto out_tensor =
      executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>();
}

std::shared_ptr<::pir::Program> BuildReduceSumProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const float value_one = 1.0;
  const std::vector<int64_t> shape = {-1, 128, 768};
  auto x = builder
               .Build<paddle::dialect::DataOp>(
                   "x", shape, phi::DataType::FLOAT32, phi::GPUPlace())
               .result(0);
  auto out = builder
                 .Build<paddle::dialect::SumOp>(
                     x, std::vector<int64_t>{-1}, phi::DataType::FLOAT32, true)
                 .result(0);
  builder.Build<paddle::dialect::FetchOp>(out, "out", 0);
  return program;
}

TEST(GroupOp, TestReduceSum) {
  // Step 1: Construct pir::Program
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  std::shared_ptr<::pir::Program> program = BuildReduceSumProgram();

  BuildAndRun(program.get());
}
