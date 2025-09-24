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

#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/drr/drr_value.h"
#include "paddle/ap/include/drr/drr_value_helper.h"
#include "paddle/ap/include/drr/ir_value.h"
#include "paddle/ap/include/drr/op_tensor_pattern_ctx_helper.h"
#include "paddle/ap/include/drr/tags.h"
#include "paddle/ap/include/drr/tensor_pattern_ctx.h"

namespace ap::drr {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnTensorPatternCtxClass();

}  // namespace ap::drr
