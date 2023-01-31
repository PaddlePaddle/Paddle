// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <map>
#include <string>
#include <type_traits>

#include "gflags/gflags.h"
#include "paddle/phi/core/macros.h"

#include "paddle/utils/variant.h"

namespace phi {

struct FlagInfo {
  using ValueType =
      paddle::variant<bool, int32_t, int64_t, uint64_t, double, std::string>;
  std::string name;
  mutable void *value_ptr;
  ValueType default_value;
  std::string doc;
  bool is_writable;
};

using ExportedFlagInfoMap = std::map<std::string, FlagInfo>;
const ExportedFlagInfoMap &GetExportedFlagInfoMap();
ExportedFlagInfoMap *GetMutableExportedFlagInfoMap();

#define __PADDLE_DEFINE_EXPORTED_FLAG(                                        \
    __name, __is_writable, __cpp_type, __gflag_type, __default_value, __doc)  \
  DEFINE_##__gflag_type(__name, __default_value, __doc);                      \
  struct __PaddleRegisterFlag_##__name {                                      \
    __PaddleRegisterFlag_##__name() {                                         \
      using FlagDeclaredType =                                                \
          typename std::remove_reference<decltype(FLAGS_##__name)>::type;     \
      static_assert(std::is_same<FlagDeclaredType, ::std::string>::value ||   \
                        std::is_arithmetic<FlagDeclaredType>::value,          \
                    "FLAGS should be std::string or arithmetic type");        \
      auto *instance = ::phi::GetMutableExportedFlagInfoMap();                \
      auto &info = (*instance)[#__name];                                      \
      info.name = #__name;                                                    \
      info.value_ptr = &(FLAGS_##__name);                                     \
      info.default_value = static_cast<__cpp_type>(__default_value);          \
      info.doc = __doc;                                                       \
      info.is_writable = __is_writable;                                       \
    }                                                                         \
    int Touch() const { return 0; }                                           \
  };                                                                          \
  static __PaddleRegisterFlag_##__name __PaddleRegisterFlag_instance##__name; \
  int TouchPaddleFlagRegister_##__name() {                                    \
    return __PaddleRegisterFlag_instance##__name.Touch();                     \
  }                                                                           \
  static_assert(std::is_same<__PaddleRegisterFlag_##__name,                   \
                             ::__PaddleRegisterFlag_##__name>::value,         \
                "FLAGS should define in global namespace")

#define PADDLE_FORCE_LINK_FLAG(__name)           \
  extern int TouchPaddleFlagRegister_##__name(); \
  UNUSED static int __paddle_use_flag_##__name = \
      TouchPaddleFlagRegister_##__name()

#define PADDLE_DEFINE_EXPORTED_bool(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, bool, bool, default_value, doc)
#define PADDLE_DEFINE_EXPORTED_READONLY_bool(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, false, bool, bool, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_int32(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, int32_t, int32, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_int64(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, int64_t, int64, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_uint64(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(                                \
      name, true, uint64_t, uint64, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_double(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, double, double, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_string(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(                                \
      name, true, ::std::string, string, default_value, doc)

}  // namespace phi
