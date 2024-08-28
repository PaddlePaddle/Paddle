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
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/fluid/pir/serialize_deserialize/include/version_compat.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/ir_printer.h"
#include "paddle/pir/include/core/op_base.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/utils.h"
#include "test/cpp/pir/tools/test_dialect.h"
#include "test/cpp/pir/tools/test_op.h"
#include "test/cpp/pir/tools/test_pir_utils.h"

TEST(save_load_version_compat, op_patch_test) {
  // (1) Init environment.
  pir::IrContext *ctx = pir::IrContext::Instance();

  // (2) Create an empty program object
  pir::Program program(ctx);
  //   pir::Program *program = new pir::Program();
  EXPECT_EQ(program.block()->empty(), true);
  const uint64_t pir_version = 0;
  pir::PatchBuilder builder(pir_version);
  builder.SetFileVersion(1);
  std::filesystem::path patch_path("/patch");
  VLOG(8) << "Patch path: " << patch_path;
  builder.BuildPatch(patch_path.string());
}
