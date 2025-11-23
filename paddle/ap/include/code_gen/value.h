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
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map.h"
#include "paddle/ap/include/axpr/dim_expr.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/code_gen/code_gen_ctx.h"
#include "paddle/ap/include/code_gen/code_gen_result.h"
#include "paddle/ap/include/code_gen/dim_expr_kernel_arg_id.h"
#include "paddle/ap/include/code_gen/in_tensor_data_ptr_kernel_arg_id.h"
#include "paddle/ap/include/code_gen/out_tensor_data_ptr_kernel_arg_id.h"
#include "paddle/ap/include/code_module/adt.h"
#include "paddle/ap/include/code_module/code_module.h"
#include "paddle/ap/include/code_module/data_type.h"
#include "paddle/ap/include/ir_match/op_match_ctx.h"
#include "paddle/ap/include/ir_match/tensor_match_ctx.h"

namespace ap::code_gen {

using axpr::Value;

}  // namespace ap::code_gen
