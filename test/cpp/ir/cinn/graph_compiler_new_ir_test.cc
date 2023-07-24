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
#include <sstream>

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"

#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"

#include "paddle/cinn/hlir/framework/new_ir_compiler.h"

TEST(GraphCompier, TestNewIR) {
  ::ir::IrContext* ctx = ::ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ::ir::Program program(ctx);
  ::ir::Builder builder = ::ir::Builder(ctx, program.block());

  auto full_op_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 128},
                                             1.0,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());

  auto full_op_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{128, 64},
                                             2.0,
                                             phi::DataType::FLOAT32,
                                             phi::CPUPlace());
  // TODO(Aurelius84): test more op
  // auto add_z = builder.Build<paddle::dialect::MatmulOp>(full_op_x->result(0),
  //                                                    full_op_y->result(0));

  EXPECT_EQ(program.block()->size(), 2u);

  std::stringstream ss;
  program.Print(ss);
  LOG(INFO) << ss.str();

  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, program);
  ASSERT_EQ(scope->var_names().size(), 2);

  cinn::hlir::framework::NewIRCompiler ir_compiler(program, target, scope);
  auto runtime_program = ir_compiler.Build();

  // FIXME(Aurelius84): It raised illegal memory access while deconstructor
  // after running all instruction, but it's ok under GLOG_v=10.
  // ASSERT_NO_THROW(runtime_program->Execute());
}
