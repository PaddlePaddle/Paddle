// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

#if defined(_WIN32)
#define PD_EXPORT_FLAG __declspec(dllexport)
#define PD_IMPORT_FLAG __declspec(dllimport)
#else
#define PD_EXPORT_FLAG
#define PD_IMPORT_FLAG
#endif  // _WIN32

// This is a simple commandline flags tool for paddle, which is inspired by
// gflags but only implements the following necessary features:
// 1. Define or declare a flag.
// 2. Parse commandline flags.
// 3. Other utility functions.

namespace paddle {
namespace flags {
/**
 * @brief Parse commandline flags.
 *
 * It recieves commandline arguments passed in argc and argv from main function,
 * argv[0] is the program name, and argv[1:] are the commandline arguments
 * which matching the format "--name=value" or "--name value". After parsing,
 * the corresponding flag value will be reset.
 */
void ParseCommandLineFlags(int* argc, char*** argv);

/**
 * @brief Allow undefined flags in ParseCommandLineFlags()
 */
void AllowUndefinedFlags();

/**
 * @brief Set flags from environment variables.
 *
 * It recieves a list of environment variable names, and set the environment
 * variable values to the corresponding flags with the same name. If error_fatal
 * is true, it will exit the program when the environment variable is not set
 * or the flag is not defined, that is the same effect as using commandline
 * argument "--fromenv=var_name1,var_name2,...". Otherwise, the errors above
 * will be ignored, that is the same effect as using commandline argument
 * "--tryfromenv=var_name1,var_name2,...".
 */
void SetFlagsFromEnv(const std::vector<std::string>& envs, bool error_fatal);

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
 * @brief Print all registered flags' current and default value.
 */
void PrintAllFlagValue();
}  // namespace flags
}  // namespace paddle

// ----------------------------DECLARE FLAGS----------------------------
#define PD_DECLARE_VARIABLE(type, name)    \
  namespace paddle {                       \
  namespace flags {                        \
  extern PD_IMPORT_FLAG type FLAGS_##name; \
  }                                        \
  }                                        \
  using paddle::flags::FLAGS_##name

#define PD_DECLARE_bool(name) PD_DECLARE_VARIABLE(bool, name)
#define PD_DECLARE_int32(name) PD_DECLARE_VARIABLE(int32_t, name)
#define PD_DECLARE_uint32(name) PD_DECLARE_VARIABLE(uint32_t, name)
#define PD_DECLARE_int64(name) PD_DECLARE_VARIABLE(int64_t, name)
#define PD_DECLARE_uint64(name) PD_DECLARE_VARIABLE(uint64_t, name)
#define PD_DECLARE_double(name) PD_DECLARE_VARIABLE(double, name)
#define PD_DECLARE_string(name) PD_DECLARE_VARIABLE(std::string, name)

namespace paddle {
namespace flags {
class FlagRegisterer {
 public:
  template <typename T>
  FlagRegisterer(std::string name,
                 std::string description,
                 std::string file,
                 const T* default_value,
                 T* value);
};
}  // namespace flags
}  // namespace paddle

// ----------------------------DEFINE FLAGS----------------------------
#define PD_DEFINE_VARIABLE(type, name, default_value, description)           \
  namespace paddle {                                                         \
  namespace flags {                                                          \
  static const type FLAGS_##name##_default = default_value;                  \
  PD_EXPORT_FLAG type FLAGS_##name = default_value;                          \
  /* Register FLAG */                                                        \
  static ::paddle::flags::FlagRegisterer flag_##name##_registerer(           \
      #name, description, __FILE__, &FLAGS_##name##_default, &FLAGS_##name); \
  }                                                                          \
  }                                                                          \
  using paddle::flags::FLAGS_##name

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
