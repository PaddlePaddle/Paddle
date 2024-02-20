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

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/sub_graph_extract_pass.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

void BuildProgram(pir::Builder &builder) {  // NOLINT
  ::pir::IrContext *ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

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
}

TEST(SubGraphExtract, softmax_test) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir::Program program(ctx);
  pir::Builder builder = pir::Builder(ctx, program.block());
  BuildProgram(builder);

  program.Print(std::cout);
  pir::PassManager pm(ctx);
  pm.AddPass(pir::CreateSubGraphExtractPass());

  pm.Run(&program);

  program.Print(std::cout);
}
