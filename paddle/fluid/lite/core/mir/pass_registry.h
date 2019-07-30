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

#pragma once

#include <string>
#include "paddle/fluid/lite/api/paddle_lite_factory_helper.h"
#include "paddle/fluid/lite/core/mir/pass_manager.h"

namespace paddle {
namespace lite {
namespace mir {

class PassRegistry {
 public:
  PassRegistry(const std::string& name, mir::Pass* pass) {
    // VLOG(2) << "Registry add MIR pass " << name;
    PassManager::Global().AddNewPass(name, pass);
  }

  bool Touch() const { return true; }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

#define REGISTER_MIR_PASS(name__, class__)                                \
  paddle::lite::mir::PassRegistry mir_pass_registry##name__(#name__,      \
                                                            new class__); \
  bool mir_pass_registry##name__##_fake() {                               \
    return mir_pass_registry##name__.Touch();                             \
  }
