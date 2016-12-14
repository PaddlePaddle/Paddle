/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#ifndef PADDLE_USE_GFLAGS
#include <stdint.h>
#include <string>
#include <vector>
#include "DisableCopy.h"

namespace paddle {

namespace flags_internal {

/**
 * Command line flag registry for special type T. It will store all command
 * arguments settings. such as name, default value.
 */
template <typename T>
struct CommandLineFlagRegistry {
  /**
   * The factory method of CommandLineFlagRegistry
   *
   * \return: The singleton instance of CommandLineFlagRegistry.
   */
  static CommandLineFlagRegistry* Instance() {
    static CommandLineFlagRegistry instance_;
    return &instance_;
  }

  struct Command {
    /// name of argument.
    std::string name;
    /// address of actual variable. such as FLAGS_xxx.
    T* value;
    /// usage text.
    std::string text;
    /// default value of this command.
    T defaultValue;
  };

  /// the command line arguments of type T.
  std::vector<Command> commands;

  DISABLE_COPY(CommandLineFlagRegistry);

private:
  inline CommandLineFlagRegistry() {}
};

/**
 *Helper class to register command line flag.
 */
template <typename T>
struct CommandLineFlagRegister {
  /**
   * \brief: Register a command line argument
   *
   * \param [in] name: The command line name.
   * \param [inout] val: The command line argument instance, FLAGS_xxx.
   * \param [in] desc: The command line helper message.
   */
  CommandLineFlagRegister(const std::string& name,
                          T* val,
                          const std::string desc) {
    CommandLineFlagRegistry<T>::Instance()->commands.push_back(
        {name, val, desc, *val});
  }
};

/**
 * \brief: Define a command line arguments.
 *
 * \param type: The variable type, such as int, double, etc.
 * \param name: The variable name. The command line argument is '--name', the
 *variable
 *is 'FLAGS_name'
 * \param default_value: The default value of command line argument.
 * \param text: The description in command line argument.
 */
#define PADDLE_DEFINE_variable(type, name, default_value, text) \
  type FLAGS_##name = default_value;                            \
  namespace paddle_flags_internal {                             \
  paddle::flags_internal::CommandLineFlagRegister<type>         \
      flags_internal_var_##name(#name, &FLAGS_##name, text);    \
  }  // namespace paddle_flags_internal

/**
 * Declare a variable to use.
 */
#define PADDLE_DECLARE_variable(type, name) extern type FLAGS_##name;

// DEFINE macro for each types.
#define P_DEFINE_int32(name, default_value, text) \
  PADDLE_DEFINE_variable(int32_t, name, default_value, text)

#define P_DEFINE_bool(name, default_value, text) \
  PADDLE_DEFINE_variable(bool, name, default_value, text)

#define P_DEFINE_string(name, default_value, text) \
  PADDLE_DEFINE_variable(std::string, name, default_value, text)

#define P_DEFINE_double(name, default_value, text) \
  PADDLE_DEFINE_variable(double, name, default_value, text)

#define P_DEFINE_int64(name, default_value, text) \
  PADDLE_DEFINE_variable(int64_t, name, default_value, text)

#define P_DEFINE_uint64(name, default_value, text) \
  PADDLE_DEFINE_variable(uint64_t, name, default_value, text)

// Declare macro for each types.
#define P_DECLARE_int32(name) PADDLE_DECLARE_variable(int32_t, name)
#define P_DECLARE_bool(name) PADDLE_DECLARE_variable(bool, name)
#define P_DECLARE_string(name) PADDLE_DECLARE_variable(std::string, name)
#define P_DECLARE_double(name) PADDLE_DECLARE_variable(double, name)
#define P_DECLARE_int64(name) PADDLE_DECLARE_variable(int64_t, name)
#define P_DECLARE_uint64(name) PADDLE_DECLARE_variable(uint64_t, name)
}  // namespace flags_internal

/**
 * \brief Parse command line flags. If parse error, just failed and exit 1.
 *
 * \param [inout] argc: The command argument count. This method will modify
 *argc, and left unused arguments.
 * \param [inout] argv: The command argument values. This method will modify
 *argv, and left unused arguments.
 * \param [in] withHelp: True will parse '-h' and '--help' to print usage.
 *
 * \note: The Command line flags format basically as follow:
 *
 *  * If the type of flag is not bool, then the follow format of command line
 *    will be parsed:
 *    * --flag_name=value
 *    * -flag_name=value
 *
 *  * If the flag is bool, then:
 *    * --flag_name=value, -flag_name=value will be parsed.
 *       * if value.tolower() == "true"| "1" will be treated as true.
 *       * else if value.tolower() == "false" | "0" will be treated as false.
 *    * --flag_name will be parsed as true.
 *    * --noflag_name will be parsed as false.
 */
void ParseCommandLineFlags(int* argc, char** argv, bool withHelp = true);

}  // namespace paddle

#else  // if use gflags.
#include <gflags/gflags.h>

#define P_DEFINE_int32 DEFINE_int32
#define P_DEFINE_bool DEFINE_bool
#define P_DEFINE_string DEFINE_string
#define P_DEFINE_double DEFINE_double
#define P_DEFINE_int64 DEFINE_int64
#define P_DEFINE_uint64 DEFINE_uint64
#define P_DECLARE_int32 DECLARE_int32
#define P_DECLARE_bool DECLARE_bool
#define P_DECLARE_string DECLARE_string
#define P_DECLARE_double DECLARE_double
#define P_DECLARE_int64 DECLARE_int64
#define P_DECLARE_uint64 DECLARE_uint64
namespace paddle {
void ParseCommandLineFlags(int* argc, char** argv, bool withHelp = true);

}  // namespace paddle

#endif
