/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/enforce.h"

#include "gflags/gflags.h"

// Note: these headers for simplify demangle type string
#include "paddle/phi/core/compat/type_defs.h"

DECLARE_int32(call_stack_level);

namespace phi {
namespace enforce {

template <typename T>
static std::string ReplaceComplexTypeStr(std::string str,
                                         const std::string& type_name) {
  auto demangle_type_str = demangle(typeid(T).name());
  size_t start_pos = 0;
  while ((start_pos = str.find(demangle_type_str, start_pos)) !=
         std::string::npos) {
    str.replace(start_pos, demangle_type_str.length(), type_name);
    start_pos += type_name.length();
  }
  return str;
}

#define __REPLACE_COMPLEX_TYPE_STR__(__TYPENAME, __STR)                      \
  do {                                                                       \
    __STR =                                                                  \
        phi::enforce::ReplaceComplexTypeStr<__TYPENAME>(__STR, #__TYPENAME); \
  } while (0)

static std::string SimplifyDemangleStr(std::string str) {
  // the older is important, you have to put complex types in front
  __REPLACE_COMPLEX_TYPE_STR__(paddle::framework::AttributeMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::framework::Attribute, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::imperative::NameVariableWrapperMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::imperative::NameVarBaseMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(std::string, str);
  return str;
}

std::string GetCurrentTraceBackString(bool for_signal = false) {
  std::ostringstream sout;

  if (!for_signal) {
    sout << "\n\n--------------------------------------\n";
    sout << "C++ Traceback (most recent call last):";
    sout << "\n--------------------------------------\n";
  }
#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
  static constexpr int TRACE_STACK_LIMIT = 100;

  void* call_stack[TRACE_STACK_LIMIT];
  auto size = backtrace(call_stack, TRACE_STACK_LIMIT);
  auto symbols = backtrace_symbols(call_stack, size);
  Dl_info info;
  int idx = 0;
  // `for_signal` used to remove the stack trace introduced by
  // obtaining the error stack trace when the signal error occurred,
  // that is not related to the signal error self, remove it to
  // avoid misleading users and developers
  int end_idx = for_signal ? 2 : 0;
  for (int i = size - 1; i >= end_idx; --i) {
    if (dladdr(call_stack[i], &info) && info.dli_sname) {
      auto demangled = demangle(info.dli_sname);
      std::string path(info.dli_fname);
      // C++ traceback info are from core.so
      if (path.substr(path.length() - 3).compare(".so") == 0) {
        sout << paddle::string::Sprintf(
            "%-3d %s\n", idx++, SimplifyDemangleStr(demangled));
      }
    }
  }
  free(symbols);
#else
  sout << "Not support stack backtrace yet.\n";
#endif
  return sout.str();
}

template <typename StrType>
static std::string GetErrorSumaryString(StrType&& what,
                                        const char* file,
                                        int line) {
  std::ostringstream sout;
  if (FLAGS_call_stack_level > 1) {
    sout << "\n----------------------\nError Message "
            "Summary:\n----------------------\n";
  }
  sout << paddle::string::Sprintf(
              "%s (at %s:%d)", std::forward<StrType>(what), file, line)
       << std::endl;
  return sout.str();
}

template <typename StrType>
static std::string GetTraceBackString(StrType&& what,
                                      const char* file,
                                      int line) {
  if (FLAGS_call_stack_level > 1) {
    // FLAGS_call_stack_level>1 means showing c++ call stack
    return GetCurrentTraceBackString() + GetErrorSumaryString(what, file, line);
  } else {
    return GetErrorSumaryString(what, file, line);
  }
}

static std::string SimplifyErrorTypeFormat(const std::string& str) {
  std::ostringstream sout;
  size_t type_end_pos = str.find(":", 0);
  if (type_end_pos == std::string::npos) {
    sout << str;
  } else {
    // Remove "Error:", add "()""
    sout << "(" << str.substr(0, type_end_pos - 5) << ")"
         << str.substr(type_end_pos + 1);
  }
  return sout.str();
}

EnforceNotMet::EnforceNotMet(std::exception_ptr e, const char* file, int line) {
  try {
    std::rethrow_exception(e);
  } catch (EnforceNotMet& e) {
    code_ = e.code();
    err_str_ = GetTraceBackString(e.what(), file, line);
    simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
  } catch (std::exception& e) {
    err_str_ = GetTraceBackString(e.what(), file, line);
    simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
  }
}

EnforceNotMet::EnforceNotMet(const std::string& str, const char* file, int line)
    : err_str_(GetTraceBackString(str, file, line)) {
  simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
}

EnforceNotMet::EnforceNotMet(const phi::ErrorSummary& error,
                             const char* file,
                             int line)
    : code_(error.code()),
      err_str_(GetTraceBackString(error.to_string(), file, line)) {
  simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
}

const char* EnforceNotMet::what() const {
  if (FLAGS_call_stack_level > 1) {
    return err_str_.c_str();
  } else {
    return simple_err_str_.c_str();
  }
}

void EnforceNotMet::set_error_str(std::string str) {
  if (FLAGS_call_stack_level > 1) {
    err_str_ = str;
  } else {
    simple_err_str_ = str;
  }
}

}  // namespace enforce
}  // namespace phi
