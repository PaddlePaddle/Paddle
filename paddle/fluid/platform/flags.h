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
#include <typeindex>
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace platform {

struct FlagInfo {
  using ValueType =
      boost::variant<bool, int32_t, int64_t, uint64_t, double, std::string>;
  std::string name;
  void *value_ptr;
  ValueType default_value;
  std::string doc;
  bool is_writable;
};

using ExportedFlagInfoMap = std::map<std::string, FlagInfo>;
const ExportedFlagInfoMap &GetExportedFlagInfoMap();

#define __PADDLE_DEFINE_EXPORTED_FLAG(__name, __is_writable, __cpp_type,    \
                                      __gflag_type, __default_value, __doc) \
  DEFINE_##__gflag_type(__name, __default_value, __doc);                    \
  struct __PaddleRegisterFlag_##__name {                                    \
    __PaddleRegisterFlag_##__name() {                                       \
      const auto &instance = ::paddle::platform::GetExportedFlagInfoMap();  \
      using Type = ::paddle::platform::ExportedFlagInfoMap;                 \
      auto &info = const_cast<Type &>(instance)[#__name];                   \
      info.name = #__name;                                                  \
      info.value_ptr = &(FLAGS_##__name);                                   \
      info.default_value = static_cast<__cpp_type>(__default_value);        \
      info.doc = __doc;                                                     \
      info.is_writable = __is_writable;                                     \
    }                                                                       \
  };                                                                        \
  static_assert(std::is_same<__PaddleRegisterFlag_##__name,                 \
                             ::__PaddleRegisterFlag_##__name>::value,       \
                "FLAGS should define in global namespace");                 \
  static __PaddleRegisterFlag_##__name __PaddleRegisterFlag_instance##__name

#define PADDLE_DEFINE_EXPORTED_bool(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, bool, bool, default_value, doc)
#define PADDLE_DEFINE_READONLY_EXPORTED_bool(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, false, bool, bool, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_int32(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, int32_t, int32, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_int64(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, int64_t, int64, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_uint64(name, default_value, doc)              \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, uint64_t, uint64, default_value, \
                                doc)

#define PADDLE_DEFINE_EXPORTED_double(name, default_value, doc) \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, double, double, default_value, doc)

#define PADDLE_DEFINE_EXPORTED_string(name, default_value, doc)    \
  __PADDLE_DEFINE_EXPORTED_FLAG(name, true, ::std::string, string, \
                                default_value, doc)

}  // namespace platform
}  // namespace paddle
