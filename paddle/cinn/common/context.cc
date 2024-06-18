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

#include "paddle/cinn/common/context.h"

#include <glog/logging.h>
#include <isl/cpp.h>

#include <mutex>

#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace common {
namespace {
#ifdef RUNTIME_INCLUDE_DIR
static constexpr char* defined_runtime_include_dir = RUNTIME_INCLUDE_DIR;
#else
static constexpr char* defined_runtime_include_dir = nullptr;
#endif
}  // namespace

thread_local isl::ctx Context::ctx_ = isl_ctx_alloc();
thread_local InfoRegistry Context::info_rgt_;
thread_local DebugManager Context::debug_mgr_;

Context& Context::Global() {
  static Context x;
  isl_options_set_on_error(ctx_.get(), ISL_ON_ERROR_ABORT);
  return x;
}

const std::vector<std::string>& Context::runtime_include_dir() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (runtime_include_dir_.empty()) {
    const char* env = std::getenv(kRuntimeIncludeDirEnvironKey);
    if (env) {  // use environment variable firstly
      VLOG(4) << "get runtime_include_dir from env: " << env;
      runtime_include_dir_ = cinn::utils::Split(env, ":");
    } else if (defined_runtime_include_dir) {
      VLOG(4) << "get runtime_include_dir from RUNTIME_INCLUDE_DIR: "
              << defined_runtime_include_dir;
      runtime_include_dir_ =
          cinn::utils::Split(defined_runtime_include_dir, ":");
    }
  }
  return runtime_include_dir_;
}

void Context::AddRuntimeIncludeDir(std::string dir) {
  // TODO(Shixiaowei02): path deduplication
  runtime_include_dir_.emplace_back(std::move(dir));
}

const char* kRuntimeIncludeDirEnvironKey = "runtime_include_dir";

std::string NameGenerator::New(const std::string& name_hint) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = name_hint_idx_.find(name_hint);
  if (it == name_hint_idx_.end()) {
    name_hint_idx_.emplace(name_hint, -1);
    return name_hint;
  }
  return name_hint + "_" + std::to_string(++it->second);
}

}  // namespace common

}  // namespace cinn
