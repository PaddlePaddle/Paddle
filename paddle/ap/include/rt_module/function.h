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
#include "paddle/ap/include/code_module/func_declare.h"
#include "paddle/ap/include/rt_module/dl_function.h"

namespace ap::rt_module {

struct FunctionImpl {
  code_module::FuncDeclare func_declare;
  DlFunction dl_function;

  bool operator==(const FunctionImpl& other) const {
    return this->func_declare == other.func_declare &&
           this->dl_function == other.dl_function;
  }
};

ADT_DEFINE_RC(Function, FunctionImpl);

}  // namespace ap::rt_module
