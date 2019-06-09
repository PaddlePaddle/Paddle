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
#include <fstream>
#include "paddle/fluid/lite/core/mir/static_kernel_pick_pass.h"
#include "paddle/fluid/lite/core/mir/type_target_transform_pass.h"
#include "paddle/fluid/lite/model_parser/model_parser.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

void Optimizer::SpecifyKernelPickTactic(core::KernelPickFactor factor) {
  auto* pass = mir::PassManager::Global().LookUp<mir::StaticKernelPickPass>(
      "static_kernel_pick_pass");
  CHECK(pass);

  *pass->mutable_kernel_pick_factors() = factor;
}

}  // namespace lite
}  // namespace paddle
