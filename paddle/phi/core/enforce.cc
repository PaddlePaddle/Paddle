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

DECLARE_int32(call_stack_level);

namespace phi {
namespace enforce {

template <typename StrType>
inline std::string GetErrorSumaryString(StrType&& what,
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
inline std::string GetTraceBackString(StrType&& what,
                                      const char* file,
                                      int line) {
  if (FLAGS_call_stack_level > 1) {
    // FLAGS_call_stack_level>1 means showing c++ call stack
    return GetCurrentTraceBackString() + GetErrorSumaryString(what, file, line);
  } else {
    return GetErrorSumaryString(what, file, line);
  }
}

const char* EnforceNotMet::what() const noexcept {
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
