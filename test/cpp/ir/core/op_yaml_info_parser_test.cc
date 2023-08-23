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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_attribute.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/op_yaml_info_parser.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"

#include "paddle/fluid/ir/dialect/paddle_dialect/interface/op_yaml_info.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"

#include "paddle/ir/core/utils.h"

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"

TEST(ir_op_info_test, op_op_info_test) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ir::Program program(ctx);

  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  ir::Builder builder(ctx, program.block());

  auto uniform1 =
      builder.Build<paddle::dialect::UniformOp>(std::vector<int64_t>{2, 2},
                                                phi::DataType::FLOAT32,
                                                0.0,
                                                1.0,
                                                2,
                                                phi::CPUPlace());

  uniform1->num_operands();
  paddle::dialect::OpYamlInfoInterface op_info_interface =
      uniform1->dyn_cast<paddle::dialect::OpYamlInfoInterface>();

  auto op_info_res = op_info_interface.GetOpInfo();

  paddle::dialect::OpYamlInfoParser op_yaml_info_parser(op_info_res);

  EXPECT_EQ((op_yaml_info_parser.AttrTypeName("seed") == "ir::Int32Attribute"),
            true);
  EXPECT_EQ(op_yaml_info_parser.IsTensorAttribute(0), true);

  EXPECT_EQ(op_yaml_info_parser.InputTensorNumber(), 0u);
}
