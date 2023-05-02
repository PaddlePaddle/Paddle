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

#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/api/ext/op_meta_info.h"

// NOTE(zengjinle): this macro is only for internal usage. Commonly, users
// should not use this macro.
#define __PD_DEFINE_RAW_OP_KERNEL_FUNC(op_name, ctx)                      \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      __reg_raw_op_kernel_func__##op_name,                                \
      "__PD_DEFINE_RAW_KERNEL_FUNC must be called in global namespace."); \
  extern "C" void PD_##op_name##_raw_op_kernel_func(                      \
      const ::paddle::framework::ExecutionContext& ctx)
