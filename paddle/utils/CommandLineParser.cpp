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

#include "CommandLineParser.h"
#ifndef PADDLE_USE_GFLAGS
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "paddle/utils/StringUtil.h"

namespace paddle {

static constexpr int kStatusOK = 0;
static constexpr int kStatusInvalid = 1;
static constexpr int kStatusNotFound = 2;

/**
 * \brief: Convert a string to any type value.
 *
 * \note: It will specialize by type T that is supported.
 */
template <typename T>
bool StringToValue(const std::string& content, T* value) {
  bool ok;
  *value = str::toWithStatus<T>(content, &ok);
  return ok;
}

template <>
bool StringToValue<bool>(const std::string& content, bool* value) {
  std::string tmp = content;

  std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](char in) -> char {
    if (in <= 'Z' && in >= 'A') {
      return in - ('Z' - 'z');
    } else {
      return in;
    }
  });  // tolower.

  if (tmp == "true" || tmp == "1") {
    *value = true;
    return true;
  } else if (tmp == "false" || tmp == "0") {
    *value = false;
    return true;
  } else {
    return false;
  }
}

template <>
bool StringToValue<std::string>(const std::string& content,
                                std::string* value) {
  *value = content;
  return true;
}

/**
 * \brief Parse argument "--blah=blah".
 *
 * \param argument: The command line argument string, such as "--blah=blah"
 * \param [out] extraInfo: The details error message for parse argument.
 * \return: kStatusOK, kStatusInvalid, kStatusNotFound
 */
template <typename T>
int ParseArgument(const std::string& argument, std::string* extraInfo) {
  for (auto& command :
       flags_internal::CommandLineFlagRegistry<T>::Instance()->commands) {
    std::string& name = command.name;
    T* value = command.value;

    std::string prefix = "--";
    prefix += name;
    prefix += "=";
    std::string content;
    if (str::startsWith(argument, prefix)) {
      content = argument.substr(prefix.size(), argument.size() - prefix.size());
    } else {
      prefix = "-";
      prefix += name;
      prefix += "=";
      if (str::startsWith(argument, prefix)) {
        content =
            argument.substr(prefix.size(), argument.size() - prefix.size());
      }
    }

    if (!content.empty()) {
      if (StringToValue(content, value)) {
        return kStatusOK;
      } else {
        *extraInfo = name;
        return kStatusInvalid;
      }
    }
  }
  return kStatusNotFound;
}

/**
 * @brief ParseBoolArgumentExtra
 * parse '--flag_name', '-flag_name' as true; '--noflag_name', '-noflag_name' as
 * false
 */
static int ParseBoolArgumentExtra(const std::string& argument,
                                  std::string* extraInfo) {
  (void)(extraInfo);  // unused extraInfo, just make api same.

  //! @warning: The order and content of prefixes is DESIGNED for parsing
  //! command line. The length of prefixes are 1, 2, 3, 4. The parse logic takes
  //! use of this fact. DO NOT CHANGE IT without reading how to parse command
  //! below.
  static const std::vector<std::pair<const char*, bool>> prefixes = {
      {"-", true}, {"--", true}, {"-no", false}, {"--no", false}};

  for (flags_internal::CommandLineFlagRegistry<bool>::Command& command :
       flags_internal::CommandLineFlagRegistry<bool>::Instance()->commands) {
    if (argument.size() > command.name.size()) {
      //! Use the length of prefix is 1, 2, 3, 4.
      size_t diff = argument.size() - command.name.size() - 1UL;
      if (diff < prefixes.size()) {
        const std::string& prefix = std::get<0>(prefixes[diff]);
        if (argument == prefix + command.name) {
          *command.value = std::get<1>(prefixes[diff]);
          return kStatusOK;
        }
      }
    }
  }
  return kStatusNotFound;
}

/**
 * \brief: Print command line arguments' usage with type T.
 */
template <typename T>
static void PrintTypeUsage() {
  for (auto& command :
       flags_internal::CommandLineFlagRegistry<T>::Instance()->commands) {
    std::string& name = command.name;
    name = "--" + name;  // Program will exit, so modify name is safe.
    std::string& desc = command.text;
    T& defaultValue = command.defaultValue;
    std::cerr << std::setw(20) << name << ": " << desc
              << "[default:" << defaultValue << "]." << std::endl;
  }
}

template <typename... TS>
static void PrintTypeUsages() {
  int unused[] = {0, (PrintTypeUsage<TS>(), 0)...};
  (void)(unused);
}
/**
 * \brief: Print all usage, and exit(1)
 */
static void PrintUsageAndExit(const char* argv0) {
  std::cerr << "Program " << argv0 << " Flags: " << std::endl;
  PrintTypeUsages<bool, int32_t, std::string, double, int64_t, uint64_t>();
  exit(1);
}

/**
 * \brief: Print the error flags, usage, and exit.
 */
static void PrintParseError(const std::string& name,
                            const char* actualInput,
                            const char* arg0) {
  std::cerr << "Parse command flag " << name << " error! User input is "
            << actualInput << std::endl;
  PrintUsageAndExit(arg0);
}

void ParseCommandLineFlags(int* argc, char** argv, bool withHelp) {
  int unused_argc = 1;
  std::string extra;
  for (int i = 1; i < *argc; ++i) {
    std::string arg = argv[i];
    int s = kStatusInvalid;
#define ParseArgumentWithType(type)           \
  s = ParseArgument<type>(arg, &extra);       \
  if (s == kStatusOK) {                       \
    continue;                                 \
  } else if (s == kStatusInvalid) {           \
    PrintParseError(extra, argv[i], argv[0]); \
  }

    ParseArgumentWithType(bool);  // NOLINT
    ParseArgumentWithType(int32_t);
    ParseArgumentWithType(double);  // NOLINT
    ParseArgumentWithType(int64_t);
    ParseArgumentWithType(uint64_t);
    ParseArgumentWithType(std::string);

#undef ParseArgumentWithType
    s = ParseBoolArgumentExtra(arg, &extra);
    if (s == kStatusOK) {
      continue;
    }

    if (withHelp && (arg == "--help" || arg == "-h")) {
      PrintUsageAndExit(argv[0]);
    }

    // NOT Found for all flags.
    std::swap(argv[unused_argc++], argv[i]);
  }
  *argc = unused_argc;
}

}  // namespace paddle
#else
namespace paddle {
#ifndef GFLAGS_NS
#define GFLAGS_NS google
#endif

namespace gflags_ns = GFLAGS_NS;

void ParseCommandLineFlags(int* argc, char** argv, bool withHelp) {
  if (withHelp) {
    gflags_ns::ParseCommandLineFlags(argc, &argv, true);
  } else {
    gflags_ns::ParseCommandLineNonHelpFlags(argc, &argv, true);
  }
}

}  // namespace paddle
#endif
