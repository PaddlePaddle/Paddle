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
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/code_module/data_type.h"
#include "paddle/ap/include/kernel_dispatch/arg_value.h"
#include "paddle/ap/include/kernel_dispatch/const_tensor.h"
#include "paddle/ap/include/kernel_dispatch/dispatch_ctx.h"
#include "paddle/ap/include/kernel_dispatch/mutable_tensor.h"
#include "paddle/ap/include/kernel_dispatch/typed_buffer.h"

namespace ap::kernel_dispatch {

using axpr::Value;

using Val = Value;

}  // namespace ap::kernel_dispatch
