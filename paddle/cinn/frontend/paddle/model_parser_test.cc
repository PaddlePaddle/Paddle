// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/paddle/model_parser.h"

#include <gtest/gtest.h>
#include "paddle/utils/flags.h"

PD_DEFINE_string(model_dir, "<NOTEXIST>", "model directory path");

namespace cinn::frontend::paddle {

TEST(LoadModelPb, naive_model) {
  hlir::framework::Scope scope;
  cpp::ProgramDesc program_desc;
  LoadModelPb(FLAGS_model_dir, "/__model__", "", &scope, &program_desc, false);

  ASSERT_EQ(program_desc.BlocksSize(), 1UL);

  auto* block = program_desc.GetBlock<cpp::BlockDesc>(0);
  ASSERT_EQ(block->OpsSize(), 4UL);
  for (int i = 0; i < block->OpsSize(); i++) {
    auto* op = block->GetOp<cpp::OpDesc>(i);
    LOG(INFO) << op->Type();
  }

  // The Op list:
  // feed
  // mul
  // scale
  // fetch
}

}  // namespace cinn::frontend::paddle
