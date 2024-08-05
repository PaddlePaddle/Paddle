Paddle / paddle / cinn / runtime / cuda /
    CMakeLists
        .txt  // Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
    using cinn::backends::GlobalSymbolRegistry;
#include "paddle/cinn/runtime/hip/hip_backend_api.h"
using cinn::runtime::hip::HIPBackendAPI;

CINN_REGISTER_HELPER(cinn_hip_host_api) {
  GlobalSymbolRegistry::Global().RegisterFn(
      "backend_api.hip", reinterpret_cast<void *>(HIPBackendAPI::Global()));
  return true;
}
