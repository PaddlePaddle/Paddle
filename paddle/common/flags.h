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

#ifdef PADDLE_WITH_GFLAGS
#include "gflags/gflags.h"
#else
#endif
#include "paddle/common/macros.h"
#include "paddle/utils/test_macros.h"
#include "paddle/utils/variant.h"

#if defined(_WIN32)
#define COMMON_IMPORT_FLAG __declspec(dllimport)
#else
#define COMMON_IMPORT_FLAG
#endif

#ifdef PADDLE_WITH_GFLAGS
#define PD_DEFINE_bool(name, val, txt) DEFINE_bool(name, val, txt)
#define PD_DEFINE_int32(name, val, txt) DEFINE_int32(name, val, txt)
#define PD_DEFINE_uint32(name, val, txt) DEFINE_uint32(name, val, txt)
#define PD_DEFINE_int64(name, val, txt) DEFINE_int64(name, val, txt)
#define PD_DEFINE_uint64(name, val, txt) DEFINE_uint64(name, val, txt)
#define PD_DEFINE_double(name, val, txt) DEFINE_double(name, val, txt)
#define PD_DEFINE_string(name, val, txt) DEFINE_string(name, val, txt)

#define PD_DECLARE_bool(name) DECLARE_bool(name)
#define PD_DECLARE_int32(name) DECLARE_int32(name)
#define PD_DECLARE_uint32(name) DECLARE_uint32(name)
#define PD_DECLARE_int64(name) DECLARE_int64(name)
#define PD_DECLARE_uint64(name) DECLARE_uint64(name)
#define PD_DECLARE_double(name) DECLARE_double(name)
#define PD_DECLARE_string(name) DECLARE_string(name)
#endif

#define PD_DECLARE_VARIABLE(type, name)     \
  namespace paddle_flags {                  \
  extern PHI_IMPORT_FLAG type FLAGS_##name; \
  }                                         \
  using paddle_flags::FLAGS_##name

#define COMMON_DECLARE_VARIABLE(type, name)    \
  namespace paddle_flags {                     \
  extern COMMON_IMPORT_FLAG type FLAGS_##name; \
  }                                            \
  using paddle_flags::FLAGS_##name

#define PD_DECLARE_bool(name) PD_DECLARE_VARIABLE(bool, name)
#define PD_DECLARE_int32(name) PD_DECLARE_VARIABLE(int32_t, name)
#define PD_DECLARE_uint32(name) PD_DECLARE_VARIABLE(uint32_t, name)
#define PD_DECLARE_int64(name) PD_DECLARE_VARIABLE(int64_t, name)
#define PD_DECLARE_uint64(name) PD_DECLARE_VARIABLE(uint64_t, name)
#define PD_DECLARE_double(name) PD_DECLARE_VARIABLE(double, name)
#define PD_DECLARE_string(name) PD_DECLARE_VARIABLE(std::string, name)

#define COMMON_DECLARE_bool(name) COMMON_DECLARE_VARIABLE(bool, name)
#define COMMON_DECLARE_int32(name) COMMON_DECLARE_VARIABLE(int32_t, name)
#define COMMON_DECLARE_uint32(name) COMMON_DECLARE_VARIABLE(uint32_t, name)
#define COMMON_DECLARE_int64(name) COMMON_DECLARE_VARIABLE(int64_t, name)
#define COMMON_DECLARE_uint64(name) COMMON_DECLARE_VARIABLE(uint64_t, name)
#define COMMON_DECLARE_double(name) COMMON_DECLARE_VARIABLE(double, name)
#define COMMON_DECLARE_string(name) COMMON_DECLARE_VARIABLE(std::string, name)
// ----------------------------DEFINE FLAGS----------------------------
#define PD_DEFINE_VARIABLE(type, name, default_value, description)           \
  namespace paddle_flags {                                                   \
  static const type FLAGS_##name##_default = default_value;                  \
  type FLAGS_##name = default_value;                                         \
  /* Register FLAG */                                                        \
  static ::paddle::flags::FlagRegisterer flag_##name##_registerer(           \
      #name, description, __FILE__, &FLAGS_##name##_default, &FLAGS_##name); \
  }                                                                          \
  using paddle_flags::FLAGS_##name

#define PD_DEFINE_bool(name, val, txt) PD_DEFINE_VARIABLE(bool, name, val, txt)
#define PD_DEFINE_int32(name, val, txt) \
  PD_DEFINE_VARIABLE(int32_t, name, val, txt)
#define PD_DEFINE_uint32(name, val, txt) \
  PD_DEFINE_VARIABLE(uint32_t, name, val, txt)
#define PD_DEFINE_int64(name, val, txt) \
  PD_DEFINE_VARIABLE(int64_t, name, val, txt)
#define PD_DEFINE_uint64(name, val, txt) \
  PD_DEFINE_VARIABLE(uint64_t, name, val, txt)
#define PD_DEFINE_double(name, val, txt) \
  PD_DEFINE_VARIABLE(double, name, val, txt)
#define PD_DEFINE_string(name, val, txt) \
  PD_DEFINE_VARIABLE(std::string, name, val, txt)

namespace paddle {
namespace flags {

/**
 * @brief Parse commandline flags.
 *
 * It receives commandline arguments passed in argc and argv from main function,
 * argv[0] is the program name, and argv[1:] are the commandline arguments
 * which matching the format "--name=value" or "--name value". After parsing,
 * the corresponding flag value will be reset.
 */
PADDLE_API void ParseCommandLineFlags(int* argc, char*** argv);

/**
 * @brief Allow undefined flags in ParseCommandLineFlags()
 */
PADDLE_API void AllowUndefinedFlags();

/**
 * @brief Set Single flag value, return true if success.
 */
bool SetFlagValue(const std::string& name, const std::string& value);

/**
 * @brief Find flag by name, return true if found.
 */
bool FindFlag(const std::string& name);

/**
 * @brief Print all registered flags' help message. If to_file is true,
 * write help message to file.
 */
void PrintAllFlagHelp(bool to_file = false,
                      const std::string& file_name = "all_flags.txt");

/**
 * @brief Get environment variable. If not found, return default value.
 */
template <typename T>
T GetFromEnv(const std::string& name, const T& default_val);

class PADDLE_API FlagRegisterer {
 public:
  template <typename T>
  FlagRegisterer(std::string name,
                 std::string description,
                 std::string file,
                 const T* default_value,
                 T* value);
};

#ifdef PADDLE_WITH_GFLAGS
inline void ParseCommandLineFlags(int* argc, char*** argv) {
  gflags::ParseCommandLineFlags(argc, argv, true);
}
#else
using paddle::flags::ParseCommandLineFlags;
#endif

#ifdef PADDLE_WITH_GFLAGS
inline bool SetFlagValue(const char* name, const char* value) {
  std::string ret = gflags::SetCommandLineOption(name, value);
  return ret.empty() ? false : true;
}
#else
using paddle::flags::SetFlagValue;
#endif

#ifdef PADDLE_WITH_GFLAGS
inline bool FindFlag(const char* name) {
  std::string value;
  return gflags::GetCommandLineOption(name, &value);
}
#else
using paddle::flags::FindFlag;
#endif

#ifdef PADDLE_WITH_GFLAGS
inline void AllowUndefinedFlags() { gflags::AllowCommandLineReparsing(); }
#else
using paddle::flags::AllowUndefinedFlags;
#endif

#ifdef PADDLE_WITH_GFLAGS
using gflags::BoolFromEnv;
using gflags::DoubleFromEnv;
using gflags::Int32FromEnv;
using gflags::Int64FromEnv;
using gflags::StringFromEnv;
using gflags::Uint32FromEnv;
using gflags::Uint64FromEnv;
#else
#define DEFINE_FROM_ENV_FUNC(type, name)                     \
  inline type name##FromEnv(const std::string& env_var_name, \
                            type default_val) {              \
    return GetFromEnv(env_var_name, default_val);            \
  }

DEFINE_FROM_ENV_FUNC(bool, Bool);
DEFINE_FROM_ENV_FUNC(int32_t, Int32);
DEFINE_FROM_ENV_FUNC(uint32_t, Uint32);
DEFINE_FROM_ENV_FUNC(int64_t, Int64);
DEFINE_FROM_ENV_FUNC(uint64_t, Uint64);
DEFINE_FROM_ENV_FUNC(double, Double);
DEFINE_FROM_ENV_FUNC(std::string, String);

#undef DEFINE_FROM_ENV_FUNC
#endif

}  // namespace flags
}  // namespace paddle

#if defined(_WIN32)
#define PHI_EXPORT_FLAG __declspec(dllexport)
#if defined(PHI_INNER)
#define PHI_IMPORT_FLAG __declspec(dllexport)
#else
#define PHI_IMPORT_FLAG __declspec(dllimport)
#endif  // PHI_INNER
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
PADDLE_API const ExportedFlagInfoMap& GetExportedFlagInfoMap();
PADDLE_API ExportedFlagInfoMap* GetMutableExportedFlagInfoMap();

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
