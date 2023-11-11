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

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>

#include "paddle/phi/core/macros.h"
#include "paddle/utils/flags.h"
#include "paddle/utils/test_macros.h"
#include "paddle/utils/variant.h"

#if defined(_WIN32)
#define PHI_EXPORT_FLAG __declspec(dllexport)
#define PHI_IMPORT_FLAG __declspec(dllimport)
#else
#define PHI_EXPORT_FLAG
#define PHI_IMPORT_FLAG
#endif  // _WIN32

// We redefine the flags macro for exporting flags defined in phi
#ifdef PADDLE_WITH_GFLAGS

// clang-format off
#define PHI_DECLARE_VARIABLE(type, shorttype, name) \
  namespace fL##shorttype {                         \
    extern PHI_IMPORT_FLAG type FLAGS_##name;       \
  }                                                 \
  using fL##shorttype::FLAGS_##name
// clang-format on

#define PHI_DEFINE_VARIABLE(type, shorttype, name, value, help) \
  namespace fL##shorttype {                                     \
    PHI_EXPORT_FLAG type FLAGS_##name = value;                  \
    static type FLAGS_no##name = value;                         \
    static GFLAGS_NAMESPACE::FlagRegisterer o_##name(           \
        #name,                                                  \
        MAYBE_STRIPPED_HELP(help),                              \
        __FILE__,                                               \
        &FLAGS_##name,                                          \
        &FLAGS_no##name);                                       \
  } /* NOLINT */                                                \
  using fL##shorttype::FLAGS_##name

#else  // PADDLE_WITH_GFLAGS

#define PHI_DECLARE_VARIABLE(type, shorttype, name) \
  namespace paddle_flags {                          \
  extern PHI_IMPORT_FLAG type FLAGS_##name;         \
  }                                                 \
  using paddle_flags::FLAGS_##name

#define PHI_DEFINE_VARIABLE(type, shorttype, name, default_value, description) \
  namespace paddle_flags {                                                     \
  static const type FLAGS_##name##_default = default_value;                    \
  PHI_EXPORT_FLAG type FLAGS_##name = default_value;                           \
  /* Register FLAG */                                                          \
  static ::paddle::flags::FlagRegisterer flag_##name##_registerer(             \
      #name, description, __FILE__, &FLAGS_##name##_default, &FLAGS_##name);   \
  }                                                                            \
  using paddle_flags::FLAGS_##name

#endif

// ----------------------------DECLARE FLAGS----------------------------
#define PHI_DECLARE_bool(name) PHI_DECLARE_VARIABLE(bool, B, name)
#define PHI_DECLARE_int32(name) PHI_DECLARE_VARIABLE(int32_t, I, name)
#define PHI_DECLARE_uint32(name) PHI_DECLARE_VARIABLE(uint32_t, U, name)
#define PHI_DECLARE_int64(name) PHI_DECLARE_VARIABLE(int64_t, I64, name)
#define PHI_DECLARE_uint64(name) PHI_DECLARE_VARIABLE(uint64_t, U64, name)
#define PHI_DECLARE_double(name) PHI_DECLARE_VARIABLE(double, D, name)

// ----------------------------DEFINE FLAGS-----------------------------
#define PHI_DEFINE_bool(name, val, txt) \
  PHI_DEFINE_VARIABLE(bool, B, name, val, txt)
#define PHI_DEFINE_int32(name, val, txt) \
  PHI_DEFINE_VARIABLE(int32_t, I, name, val, txt)
#define PHI_DEFINE_uint32(name, val, txt) \
  PHI_DEFINE_VARIABLE(uint32_t, U, name, val, txt)
#define PHI_DEFINE_int64(name, val, txt) \
  PHI_DEFINE_VARIABLE(int64_t, I64, name, val, txt)
#define PHI_DEFINE_uint64(name, val, txt) \
  PHI_DEFINE_VARIABLE(uint64_t, U64, name, val, txt)
#define PHI_DEFINE_double(name, val, txt) \
  PHI_DEFINE_VARIABLE(double, D, name, val, txt)

#ifdef PADDLE_WITH_GFLAGS
#define PHI_DECLARE_string(name)                        \
  namespace fLS {                                       \
  extern PHI_IMPORT_FLAG ::fLS::clstring& FLAGS_##name; \
  }                                                     \
  using fLS::FLAGS_##name

#define PHI_DEFINE_string(name, val, txt)                                    \
  namespace fLS {                                                            \
  using ::fLS::clstring;                                                     \
  clstring FLAGS_##name##_default = val;                                     \
  clstring FLAGS_##name##_current = val;                                     \
  static GFLAGS_NAMESPACE::FlagRegisterer o_##name(#name,                    \
                                                   MAYBE_STRIPPED_HELP(txt), \
                                                   __FILE__,                 \
                                                   &FLAGS_##name##_current,  \
                                                   &FLAGS_##name##_default); \
  PHI_EXPORT_FLAG clstring& FLAGS_##name = FLAGS_##name##_current;           \
  } /* NOLINT */                                                             \
  using ::fLS::FLAGS_##name
#else
#define PHI_DECLARE_string(name) PHI_DECLARE_VARIABLE(std::string, S, name)

#define PHI_DEFINE_string(name, val, txt) \
  PHI_DEFINE_VARIABLE(std::string, S, name, val, txt)
#endif

namespace phi {

struct FlagInfo {
  using ValueType =
      paddle::variant<bool, int32_t, int64_t, uint64_t, double, std::string>;
  std::string name;
  mutable void* value_ptr;
  ValueType default_value;
  std::string doc;
  bool is_writable;
};

using ExportedFlagInfoMap = std::map<std::string, FlagInfo>;
TEST_API const ExportedFlagInfoMap& GetExportedFlagInfoMap();
TEST_API ExportedFlagInfoMap* GetMutableExportedFlagInfoMap();

#define __PHI_DEFINE_EXPORTED_FLAG(                                           \
    __name, __is_writable, __cpp_type, __gflag_type, __default_value, __doc)  \
  PHI_DEFINE_##__gflag_type(__name, __default_value, __doc);                  \
  struct __PaddleRegisterFlag_##__name {                                      \
    __PaddleRegisterFlag_##__name() {                                         \
      using FlagDeclaredType =                                                \
          typename std::remove_reference<decltype(FLAGS_##__name)>::type;     \
      static_assert(std::is_same<FlagDeclaredType, ::std::string>::value ||   \
                        std::is_arithmetic<FlagDeclaredType>::value,          \
                    "FLAGS should be std::string or arithmetic type");        \
      auto* instance = ::phi::GetMutableExportedFlagInfoMap();                \
      auto& info = (*instance)[#__name];                                      \
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

#define PHI_DEFINE_EXPORTED_bool(name, default_value, doc) \
  __PHI_DEFINE_EXPORTED_FLAG(name, true, bool, bool, default_value, doc)
#define PHI_DEFINE_EXPORTED_READONLY_bool(name, default_value, doc) \
  __PHI_DEFINE_EXPORTED_FLAG(name, false, bool, bool, default_value, doc)

#define PHI_DEFINE_EXPORTED_int32(name, default_value, doc) \
  __PHI_DEFINE_EXPORTED_FLAG(name, true, int32_t, int32, default_value, doc)

#define PHI_DEFINE_EXPORTED_int64(name, default_value, doc) \
  __PHI_DEFINE_EXPORTED_FLAG(name, true, int64_t, int64, default_value, doc)

#define PHI_DEFINE_EXPORTED_uint64(name, default_value, doc) \
  __PHI_DEFINE_EXPORTED_FLAG(name, true, uint64_t, uint64, default_value, doc)

#define PHI_DEFINE_EXPORTED_double(name, default_value, doc) \
  __PHI_DEFINE_EXPORTED_FLAG(name, true, double, double, default_value, doc)

#define PHI_DEFINE_EXPORTED_string(name, default_value, doc) \
  __PHI_DEFINE_EXPORTED_FLAG(                                \
      name, true, ::std::string, string, default_value, doc)

}  // namespace phi
