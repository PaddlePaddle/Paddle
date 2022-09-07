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

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "paddle/utils/blank.h"
#include "paddle/utils/variant.h"

namespace egr {
class EagerVariable;
}
namespace paddle {
namespace framework {
class VarDesc;
class BlockDesc;
using Attribute = paddle::variant<paddle::blank,
                                  int,
                                  float,
                                  std::string,
                                  std::vector<int>,
                                  std::vector<float>,
                                  std::vector<std::string>,
                                  bool,
                                  std::vector<bool>,
                                  BlockDesc*,
                                  int64_t,
                                  std::vector<BlockDesc*>,
                                  std::vector<int64_t>,
                                  std::vector<double>,
                                  VarDesc*,
                                  std::vector<VarDesc*>,
                                  double>;
using AttributeMap = std::unordered_map<std::string, Attribute>;
}  // namespace framework
namespace imperative {
class VariableWrapper;
class SavedVariableWrapperList;
class VarBase;

namespace details {
template <typename T>
struct NameVarMapTrait {};

template <>
struct NameVarMapTrait<VarBase> {
  using Type = std::map<std::string, std::vector<std::shared_ptr<VarBase>>>;
};

template <>
struct NameVarMapTrait<VariableWrapper> {
  using Type = std::map<std::string, SavedVariableWrapperList>;
};

template <>
struct NameVarMapTrait<egr::EagerVariable> {
  using Type =
      std::map<std::string, std::vector<std::shared_ptr<egr::EagerVariable>>>;
};

}  // namespace details

template <typename T>
using NameVarMap = typename details::NameVarMapTrait<T>::Type;

using NameVarBaseMap = NameVarMap<VarBase>;
using NameVariableWrapperMap = NameVarMap<VariableWrapper>;
using NameTensorMap = NameVarMap<egr::EagerVariable>;

}  // namespace imperative
}  // namespace paddle

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
  __REPLACE_COMPLEX_TYPE_STR__(paddle::imperative::NameTensorMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(std::string, str);
  return str;
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

std::string SimplifyErrorTypeFormat(const std::string& str) {
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

}  // namespace enforce
}  // namespace phi
