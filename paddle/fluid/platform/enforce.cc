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

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/debug_support.h"

namespace paddle {
namespace platform {

template <typename StrType>
inline void EnforceNotMet::Init(StrType what, const char* f, int l) {
  static constexpr int TRACE_STACK_LIMIT = 100;
  std::ostringstream sout;

  sout << "C++ Callstacks: \n";
  sout << string::Sprintf("%s at [%s:%d]", what, f, l) << std::endl;
#if !defined(_WIN32)
  void* call_stack[TRACE_STACK_LIMIT];
  auto size = backtrace(call_stack, TRACE_STACK_LIMIT);
  auto symbols = backtrace_symbols(call_stack, size);
  Dl_info info;
  for (int i = 0; i < size; ++i) {
    if (dladdr(call_stack[i], &info) && info.dli_sname) {
      auto demangled = demangle(info.dli_sname);
      auto addr_offset = static_cast<char*>(call_stack[i]) -
                         static_cast<char*>(info.dli_saddr);
      sout << string::Sprintf("%-3d %*0p %s + %zd\n", i, 2 + sizeof(void*) * 2,
                              call_stack[i], demangled, addr_offset);
    } else {
      sout << string::Sprintf("%-3d %*0p\n", i, 2 + sizeof(void*) * 2,
                              call_stack[i]);
    }
  }
  free(symbols);
#else
  sout << "Windows not support stack backtrace yet.";
#endif
  err_str_ += sout.str();
}

template <>
inline void EnforceNotMet::Init(std::vector<std::string> what, const char* f,
                                int l) {
  std::ostringstream sout;
  sout << "\nPython Callstacks: \n";
  for (auto& line : what) {
    sout << line;
  }
  err_str_ += sout.str();
}

EnforceNotMet::EnforceNotMet(std::exception_ptr e, const char* f, int l) {
  try {
    std::rethrow_exception(e);
  } catch (std::exception& e) {
    auto callstack = platform::PythonDebugSupport::GetInstance()->get();
    Init(callstack, f, l);
    Init(e.what(), f, l);
  }
}

template <typename... ARGS>
EnforceNotMet::EnforceNotMet(const char* f, int l, ARGS... args) {
  Init(string::Sprintf(args...), f, l);
}

}  // namespace platform
}  // namespace paddle
