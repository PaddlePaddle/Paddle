/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/init_phi.h"

#include "paddle/common/macros.h"
#include "paddle/fluid/platform/init.h"

#include "paddle/fluid/prim/utils/static/static_tensor_operants.h"
#include "paddle/phi/api/include/operants_manager.h"

REGISTER_FILE_SYMBOLS(init_phi)

namespace paddle {

InitPhi::InitPhi() { paddle::framework::InitMemoryMethod(); }

void InitOperants() {
  paddle::OperantsManager::Instance().static_operants =
      std::make_unique<paddle::prim::StaticTensorOperants>();
}

}  // namespace paddle
