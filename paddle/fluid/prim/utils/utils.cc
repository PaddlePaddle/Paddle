// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/phi/core/flags.h"

PADDLE_DEFINE_EXPORTED_bool(prim_enabled, false, "enable_prim or not");
namespace paddle {
namespace prim {
bool PrimCommonUtils::IsPrimEnabled() {
  return StaticCompositeContext::Instance().IsPrimEnabled();
}

void PrimCommonUtils::SetPrimEnabled(bool enable_prim) {
  return StaticCompositeContext::Instance().SetPrimEnabled(enable_prim);
}
}  // namespace prim
}  // namespace paddle
