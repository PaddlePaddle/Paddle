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

#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"

#include <absl/strings/string_view.h>
#include <glog/raw_logging.h>

#include <iostream>

#include "paddle/cinn/runtime/flags.h"
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
PD_DECLARE_bool(verbose_function_register);

namespace cinn {
namespace backends {

RuntimeSymbols &GlobalSymbolRegistry::Global() {
  static RuntimeSymbols symbols;
  return symbols;
}

void *RuntimeSymbols::Lookup(absl::string_view name) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = symbols_.find(std::string(name));
  if (it != symbols_.end()) {
    return it->second;
  }

  return nullptr;
}

void RuntimeSymbols::Register(const std::string &name, void *address) {
#ifdef CINN_WITH_DEBUG
  if (FLAGS_verbose_function_register) {
    RAW_LOG_INFO("JIT Register function [%s]: %p", name.c_str(), address);
  }
#endif  // CINN_WITH_DEBUG
  std::lock_guard<std::mutex> lock(mu_);
  auto it = symbols_.find(name);
  if (it != symbols_.end()) {
    PADDLE_ENFORCE_EQ(
        it->second,
        address,
        ::common::errors::InvalidArgument("Duplicate register symbol"));
    return;
  }

  symbols_.insert({name, reinterpret_cast<void *>(address)});
}

void RuntimeSymbols::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  symbols_.clear();
  scalar_holder_.clear();
}

}  // namespace backends
}  // namespace cinn
