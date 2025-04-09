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

#include "paddle/cinn/runtime/backend_api.h"
#include <glog/logging.h>
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/common/arch_util.h"
#include "paddle/common/enforce.h"

using cinn::backends::GlobalSymbolRegistry;

namespace cinn {
namespace runtime {

BackendAPI* BackendAPI::get_backend(common::Arch arch) {
  void* temp_backend_api;
  arch.Match(
      [&](common::HygonDCUArchHIP) {
        temp_backend_api =
            GlobalSymbolRegistry::Global().Lookup("backend_api.hip");
        PADDLE_ENFORCE_NE(temp_backend_api,
                          nullptr,
                          ::common::errors::InvalidArgument(
                              "global symbol (backend_api.hip) not found!"));
      },
      [&](common::HygonDCUArchSYCL) {
        temp_backend_api =
            GlobalSymbolRegistry::Global().Lookup("backend_api.sycl");
        PADDLE_ENFORCE_NE(temp_backend_api,
                          nullptr,
                          ::common::errors::InvalidArgument(
                              "global symbol (backend_api.sycl) not found!"));
      },
      [&](std::variant<common::UnknownArch,
                       common::X86Arch,
                       common::ARMArch,
                       common::NVGPUArch>) {
        std::stringstream ss;
        ss << "Target(" << arch << ") is not support get_backend now.";
        PADDLE_THROW(::common::errors::Fatal(ss.str()));
      });
  return reinterpret_cast<BackendAPI*>(temp_backend_api);
}

BackendAPI* BackendAPI::get_backend(const common::Target target) {
  return get_backend(target.arch);
}

}  // namespace runtime
}  // namespace cinn
