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
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/cast_utils.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/op_info.h"
#include "paddle/ir/core/parameter.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/value.h"
#include "paddle/ir/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"
#include "paddle/phi/core/kernel_registry.h"

TEST(pattern_rewrite, Patterns) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ir::Program program(ctx);

  ir::PassManager pm(ctx);
  pm.AddPass(ir::CreateShapeOptimizationPass());
  CHECK_EQ(pm.Run(&program), true);
}
