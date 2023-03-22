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

#include "paddle/fluid/prim/tests/init_env_utils.h"
#include "paddle/fluid/prim/utils/eager/eager_tensor_operants.h"
#include "paddle/fluid/prim/utils/static/static_tensor_operants.h"
#include "paddle/phi/api/include/operants_manager.h"

namespace paddle {
namespace prim {

void InitTensorOperants() {
  paddle::OperantsManager::Instance().eager_operants.reset(
      new paddle::prim::EagerTensorOperants());
  paddle::OperantsManager::Instance().static_operants.reset(
      new paddle::prim::StaticTensorOperants());
}

}  // namespace prim
}  // namespace paddle
