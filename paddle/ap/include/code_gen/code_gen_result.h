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

#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map.h"
#include "paddle/ap/include/axpr/core_expr.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/code_module/code_module.h"

namespace ap::code_gen {

template <typename ValueT>
struct CodeGenResultImpl {
  code_module::CodeModule code_module;
  axpr::Function<axpr::SerializableValue> kernel_dispatch_func;
  axpr::AttrMap<axpr::SerializableValue> kernel_dispatch_const_data;

  bool operator==(const CodeGenResultImpl& other) const {
    return this == &other;
  }
};

template <typename ValueT>
ADT_DEFINE_RC(CodeGenResult, CodeGenResultImpl<ValueT>);

}  // namespace ap::code_gen
