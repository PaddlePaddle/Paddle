/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/common/enforce.h"
#include <array>
#include <atomic>
#include <map>
#include <string>
#include <vector>
#include "paddle/common/flags.h"

REGISTER_LOG_SIMPLY_STR(std::string);
COMMON_DECLARE_int32(call_stack_level);
namespace {
class StrSizeCmp {
 public:
  bool operator()(const std::string& lhs, const std::string& rhs) const {
    return lhs.size() > rhs.size();
  }
};

using LogSimplyStrMap = std::map<std::string, std::string, StrSizeCmp>;

LogSimplyStrMap& GetLogStrSimplyMap() {
  static LogSimplyStrMap str_simply_map;
  return str_simply_map;
}

std::string SimplifyDemangleStr(std::string str) {
  auto& str_map = GetLogStrSimplyMap();
  for (auto& value : str_map) {
    size_t start_pos = 0;
    while ((start_pos = str.find(value.first, start_pos)) !=
           std::string::npos) {
      str.replace(start_pos, value.first.length(), value.second);
      start_pos += value.second.length();
    }
  }
  return str;
}

std::atomic_bool paddle_fatal_skip{false};

}  // namespace

namespace common::enforce {
void SkipPaddleFatal(bool skip) { paddle_fatal_skip.store(skip); }
bool IsPaddleFatalSkip() { return paddle_fatal_skip.load(); }

int GetCallStackLevel() { return FLAGS_call_stack_level; }

std::string SimplifyErrorTypeFormat(const std::string& str) {
  std::ostringstream sout;
  size_t type_end_pos = str.find(':', 0);
  if (type_end_pos != str.npos && type_end_pos >= 5 &&
      str.substr(type_end_pos - 5, 6) == "Error:") {
    // Remove "Error:", add "()"
    // Examples:
    //    InvalidArgumentError: xxx -> (InvalidArgument) xxx
    sout << "(" << str.substr(0, type_end_pos - 5) << ")"
         << str.substr(type_end_pos + 1);
  } else {
    // type_end_pos == std::string::npos
    sout << str;
  }
  return sout.str();
}
bool RegisterLogSimplyStr(const std::string& type_name,
                          const std::string& simply_name) {
  return GetLogStrSimplyMap()
      .emplace(std::make_pair(type_name, simply_name))
      .second;
}

std::string GetCurrentTraceBackString(bool for_signal) {
  std::ostringstream sout;

  if (!for_signal) {
    sout << "\n\n--------------------------------------\n";
    sout << "C++ Traceback (most recent call last):";
    sout << "\n--------------------------------------\n";
  }
#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
  static constexpr int TRACE_STACK_LIMIT = 100;

  std::array<void*, TRACE_STACK_LIMIT> call_stack = {};
  auto size = backtrace(call_stack.data(), TRACE_STACK_LIMIT);
  auto symbols = backtrace_symbols(call_stack.data(), size);
  Dl_info info;
  int idx = 0;
  // `for_signal` used to remove the stack trace introduced by
  // obtaining the error stack trace when the signal error occurred,
  // that is not related to the signal error self, remove it to
  // avoid misleading users and developers
  int end_idx = for_signal ? 2 : 0;
  for (int i = size - 1; i >= end_idx; --i) {
    if (dladdr(call_stack[i], &info) && info.dli_sname) {
      auto demangled = common::demangle(info.dli_sname);
      std::string path(info.dli_fname);
      // C++ traceback info are from core.so
      if (path.substr(path.length() - 3) == ".so") {
        sout << paddle::string::Sprintf(
            "%-3d %s\n", idx++, SimplifyDemangleStr(demangled));
      }
    }
  }
  free(symbols);  // NOLINT
#else
  sout << "Not support stack backtrace yet.\n";
#endif
  return sout.str();
}

}  // namespace common::enforce
