// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/backends/function_prototype.h"

namespace cinn {
namespace backends {

static const char* extern_func__tanh_v = "tanh_v";

class ExternFunctionProtoRegistry : public FunctionProtoRegistry {
 public:
  using FunctionProtoRegistry::Lookup;
  using FunctionProtoRegistry::Register;

  static ExternFunctionProtoRegistry& Global();

 private:
  ExternFunctionProtoRegistry();
  CINN_DISALLOW_COPY_AND_ASSIGN(ExternFunctionProtoRegistry);
};

namespace detail {

FunctionProto* CreateTanhVProto();

}  // namespace detail

}  // namespace backends
}  // namespace cinn
