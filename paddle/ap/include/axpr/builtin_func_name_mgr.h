// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include <unordered_map>
#include "glog/logging.h"

namespace ap::axpr {

struct BuiltinFuncName {
  std::optional<std::string> module_name{};
  std::string func_name;

  std::string ToString() const {
    return OptStrToStr(module_name) + "." + func_name;
  }

  static std::string OptStrToStr(const std::optional<std::string>& opt_str) {
    if (opt_str.has_value()) return opt_str.value();
    return "__builtin_frame__";
  }
};

class BuiltinFuncNameMgr {
 public:
  bool Has(void* ptr) const { return func_ptr2name_.count(ptr) > 0; }

  std::optional<BuiltinFuncName> OptGet(void* ptr) const {
    const auto& iter = func_ptr2name_.find(ptr);
    if (iter == func_ptr2name_.end()) return std::nullopt;
    return iter->second;
  }

  void Register(const std::optional<std::string>& module_name,
                const std::string& func_name,
                void* func_ptr) {
    CHECK(func_ptr2name_
              .emplace(func_ptr, BuiltinFuncName{module_name, func_name})
              .second)
        << "redundant name for builtin function: old_module_name: "
        << ToString(func_ptr2name_[func_ptr].module_name)
        << ", old_func_name: " << func_ptr2name_[func_ptr].func_name
        << ", new_module_name: " << ToString(module_name)
        << ", new_func_name: " << func_name;
  }

  static BuiltinFuncNameMgr* Singleton() {
    static BuiltinFuncNameMgr mgr{};
    return &mgr;
  }

 private:
  BuiltinFuncNameMgr() {}

  static std::string ToString(const std::optional<std::string>& opt_str) {
    return BuiltinFuncName::OptStrToStr(opt_str);
  }

  std::unordered_map<void*, BuiltinFuncName> func_ptr2name_;
};

}  // namespace ap::axpr
