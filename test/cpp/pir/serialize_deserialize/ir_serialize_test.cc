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
#include <gtest/gtest.h>
#include <chrono>
#include <sstream>

#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/program.h"

using namespace paddle::dialect;  // NOLINT

TEST(op_test, base) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 512, 64},
                                             1.5);
  // linear 1
  paddle::dialect::FullOp full_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64}, 1.5);
  paddle::dialect::FullOp full_bias_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);
  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                               full_weight_op1.out());
  builder.Build<paddle::dialect::AddOp>(
      matmul_op1.out(), full_bias_op1.out());
  std::string file_path = "./test_serialize_2.json";
  uint64_t version = 1;
  WriteModule(program, file_path, version, true, true);
}

TEST(op_test, time_test) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  pir::Program program(ctx);
  pir::Block* block = program.block();
  pir::Builder builder(ctx, block);

  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 512, 64},
                                             1.5);

  for (int i = 0; i < 2500; ++i) {
    paddle::dialect::FullOp full_weight_op =
        builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64},
                                               1.5);
    paddle::dialect::FullOp full_bias_op =
        builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);

    paddle::dialect::MatmulOp matmul_op =
        builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                                 full_weight_op.out());
    builder.Build<paddle::dialect::AddOp>(
        matmul_op.out(), full_bias_op.out());
  }

  std::string file_path = "./test_serialize_10000_dump0.json";
  uint64_t version = 1;
  auto start = std::chrono::high_resolution_clock::now();
  WriteModule(program, file_path, version, true, false);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // 输出时间差
  std::cout << "Elapsed time: " << duration.count() << " microseconds"
            << std::endl;
}
