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
#include <absl/types/any.h>
#include <isl/cpp.h>

#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "paddle/cinn/common/debug_manager.h"
#include "paddle/cinn/common/info_registry.h"
#include "paddle/cinn/common/target.h"
#include "paddle/utils/flags.h"

namespace cinn {

PD_DECLARE_bool(cinn_runtime_display_debug_info);

namespace ir {
class Expr;
}  // namespace ir

namespace common {

extern const char* kRuntimeIncludeDirEnvironKey;

struct NameGenerator {
  std::string New(const std::string& name_hint);

  // Reset id to initial.
  void ResetID() {
    std::lock_guard<std::mutex> lock(mutex_);
    name_hint_idx_.clear();
  }

 private:
  absl::flat_hash_map<std::string, uint32_t> name_hint_idx_;
  mutable std::mutex mutex_;
};

class Context {
 public:
  static Context& Global();

  /**
   * Generate a new unique name.
   * @param name_hint The prefix.
   */
  std::string NewName(const std::string& name_hint) {
    return name_generator_.New(name_hint);
  }

  void ResetNameId() { name_generator_.ResetID(); }

  const std::vector<std::string>& runtime_include_dir();

  void AddRuntimeIncludeDir(std::string dir);

  /**
   * The global isl ctx.
   */
  static isl::ctx& isl_ctx() { return ctx_; }

  static InfoRegistry& info_rgt() { return info_rgt_; }

  static DebugManager& debug_mgr() { return debug_mgr_; }

 private:
  Context() = default;

  NameGenerator name_generator_;
  std::vector<std::string> runtime_include_dir_;
  mutable std::mutex mutex_;

  static thread_local isl::ctx ctx_;
  static thread_local InfoRegistry info_rgt_;
  static thread_local DebugManager debug_mgr_;
};

static std::string UniqName(const std::string& prefix) {
  return Context::Global().NewName(prefix);
}

}  // namespace common
}  // namespace cinn
