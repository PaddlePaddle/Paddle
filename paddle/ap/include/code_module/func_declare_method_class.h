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
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/code_module/func_declare.h"

namespace ap::code_module {

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetFuncDeclareClass();

}  // namespace ap::code_module
