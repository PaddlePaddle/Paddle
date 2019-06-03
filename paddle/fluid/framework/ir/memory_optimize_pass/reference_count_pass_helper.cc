// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {
namespace ir {

VarDesc *TryGetLatestVarDesc(const std::vector<details::VarHandle *> &vars) {
  VarDesc *var_desc = nullptr;
  std::find_if(vars.rbegin(), vars.rend(),
               [&](details::VarHandle *var_handle) -> bool {
                 var_desc = var_handle->Node()->Var();
                 return var_desc != nullptr;
               });
  return var_desc;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
