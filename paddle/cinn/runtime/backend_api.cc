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
#include "paddle/cinn/runtime/flags.h"
#include "paddle/common/enforce.h"

using cinn::backends::GlobalSymbolRegistry;

namespace cinn {
namespace runtime {
BackendAPI* BackendAPI::get_backend(const common::Target target) {
  return get_backend(target.language);
}

BackendAPI* BackendAPI::get_backend(common::Target::Language target_language) {
  CheckCompileWith(target_language);
  void* temp_backend_api;
  switch (target_language) {
    case common::Target::Language::cuda:
      temp_backend_api =
          GlobalSymbolRegistry::Global().Lookup("backend_api.cuda");
      PADDLE_ENFORCE_NOT_NULL(
          temp_backend_api,
          ::common::errors::Fatal(
              "global symbol (backend_api.cuda) not found!"));
      break;
    case common::Target::Language::sycl:
      temp_backend_api =
          GlobalSymbolRegistry::Global().Lookup("backend_api.sycl");
      PADDLE_ENFORCE_NOT_NULL(
          temp_backend_api,
          ::common::errors::Fatal(
              "global symbol (backend_api.sycl) not found!"));
      break;
    case common::Target::Language::hip:
      temp_backend_api =
          GlobalSymbolRegistry::Global().Lookup("backend_api.hip");
      PADDLE_ENFORCE_NOT_NULL(
          temp_backend_api,
          ::common::errors::Fatal(
              "global symbol (backend_api.hip) not found!"));
      break;
    case common::Target::Language::bangc:
      PADDLE_THROW(::common::errors::Fatal(
          "Target(bangc) is not support get_backend now."));
    default:
      std::ostringstream error_message;
      error_message << "Target(" << target_language
                    << ") is not supported now.";
      PADDLE_THROW(::common::errors::Fatal(error_message.str()));
  }
  return reinterpret_cast<BackendAPI*>(temp_backend_api);
}

}  // namespace runtime
}  // namespace cinn
