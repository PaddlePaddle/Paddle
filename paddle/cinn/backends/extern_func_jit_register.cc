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

#include "paddle/cinn/backends/extern_func_jit_register.h"

#include <string>

namespace cinn {
namespace backends {

void RegisterExternFunctionHelper(const std::string &fn_name,
                                  std::unique_ptr<FunctionProto> &&fn_proto,
                                  Target target,
                                  void *fn_ptr) {
  ExternFunctionProtoRegistry::Global().Register(fn_name, fn_proto.release());
  PADDLE_ENFORCE_NOT_NULL(ExternFunctionProtoRegistry::Global().Lookup(fn_name),
                          ::common::errors::NotFound(
                              "The function prototype for '%s' was not found "
                              "in the ExternFunctionProtoRegistry. Please "
                              "ensure the function name is correct.",
                              fn_name));

  ExternFunctionEmitterRegistry::Global().Register(
      ExternFuncID{TargetToBackendRepr(target), fn_name.c_str()}, fn_name);

  GlobalSymbolRegistry::Global().RegisterFn(fn_name,
                                            reinterpret_cast<void *>(fn_ptr));
}

void RegisterExternFunction::End() {
  auto fn_proto = fn_proto_builder_.Build();
  RegisterExternFunctionHelper(fn_name_, std::move(fn_proto), target_, fn_ptr_);
}

}  // namespace backends
}  // namespace cinn
