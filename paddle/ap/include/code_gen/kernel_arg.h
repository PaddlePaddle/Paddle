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

#pragma once
#include "paddle/ap/include/axpr/attr_map.h"
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/code_gen/kernel_arg_id.h"
#include "paddle/ap/include/code_module/adt.h"

namespace ap::code_gen {

struct KernelArgImpl {
  KernelArgId kernel_arg_id;
  axpr::Lambda<axpr::CoreExpr> getter_lambda;

  bool operator==(const KernelArgImpl& other) const {
    return this->kernel_arg_id == other.kernel_arg_id &&
           this->getter_lambda == other.getter_lambda;
  }
};

ADT_DEFINE_RC(KernelArg, KernelArgImpl);

}  // namespace ap::code_gen
