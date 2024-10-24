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

#include <gtest/gtest.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "paddle/common/layout.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/general/auto_layout_pass.h"
#include "paddle/fluid/pir/transforms/general/auto_layout_simplify_pass.h"
#include "paddle/fluid/pir/transforms/general/constant_folding_pass.h"
#include "paddle/fluid/pir/transforms/passes.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/parser/ir_parser.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass_manager.h"
using pir::IrParser;

TEST(auto_layout_pass, pass_test) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();

  std::stringstream ss;
  std::ifstream file("auto_layout_program.txt");
  if (file) {
    ss << file.rdbuf();
  }

  auto program = pir::IrParser(ctx, ss).ParseProgram();

  pir::PassManager auto_layout_pm(::pir::IrContext::Instance(), 3);
  auto_layout_pm.AddPass(pir::CreateAutoLayoutPass());
  auto_layout_pm.AddPass(pir::CreateAutoLayoutSimplifyPass());
  auto_layout_pm.Run(program.get());
}
