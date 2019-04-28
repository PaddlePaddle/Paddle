// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/optimizer.h"
#include <gtest/gtest.h>
#include "paddle/fluid/lite/core/mir/generate_program_pass.h"
#include "paddle/fluid/lite/core/mir/pass_manager.h"
#include "paddle/fluid/lite/core/mir/passes.h"
#include "paddle/fluid/lite/core/mir/static_kernel_pick_pass.h"
#include "paddle/fluid/lite/core/program_fake_utils.h"

namespace paddle {
namespace lite {

TEST(Optimizer, test) {
  Optimizer optimizer;
  auto program = ProgramFaker();
  std::vector<Place> places({Place{TARGET(kHost), PRECISION(kFloat)}});

  auto* pick_pass =
      mir::PassManager::Global().LookUp<mir::StaticKernelPickPass>(
          "static_kernel_pick_pass");
  ASSERT_TRUE(pick_pass != nullptr);
  pick_pass->mutable_kernel_pick_factors()
      ->ConsiderTarget()
      .ConsiderPrecision();

  optimizer.Run(std::move(program), places);
  auto runtime_program = optimizer.GenRuntimeProgram();
  LOG(INFO) << "num instructions " << runtime_program->num_instructions();
}

}  // namespace lite
}  // namespace paddle

USE_LITE_OP(fc);
USE_LITE_KERNEL(fc, kHost, kFloat, def);
