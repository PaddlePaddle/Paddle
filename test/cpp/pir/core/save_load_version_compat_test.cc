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
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/serialize_deserialize/include/version_compat.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/utils.h"

TEST(save_load_version_compat, op_patch_test) {
  // (1) Init environment.
  pir::IrContext *ctx = pir::IrContext::Instance();

  // (2) Create an empty program object
  pir::Program program(ctx);
  //   pir::Program *program = new pir::Program();
  EXPECT_EQ(program.block()->empty(), true);
  const uint64_t pir_version = 2;
  pir::PatchBuilder builder(pir_version);
  std::string cur_file = std::string(__FILE__);
  std::string yaml_file =
      cur_file.substr(0, cur_file.rfind('/')) + "/patch.yaml";
  builder.BuildPatch(yaml_file);
}
