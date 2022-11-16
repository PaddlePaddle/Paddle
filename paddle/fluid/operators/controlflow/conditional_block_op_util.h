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

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace operators {

static const char ConditionalOp_kInputs[] = "Input";
static const char ConditionalOp_kOutputs[] = "Out";
static const char ConditionalOp_kCondition[] = "Cond";
static const char ConditionalOp_kScope[] = "Scope";
static const char ConditionalOp_kSkipEagerDeletionVars[] =
    "skip_eager_deletion_vars";

}  // namespace operators
}  // namespace paddle
