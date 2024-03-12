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

#include "paddle/common/flags.h"

#include <cstdlib>
#include "gtest/gtest.h"

PD_DEFINE_int32(paddle_test_int32, 1, "test int32 flag");
PD_DEFINE_uint32(paddle_test_uint32, 2, "test uint32 flag");
PD_DEFINE_string(paddle_test_string, "raw", "test string flag");

using namespace paddle::flags;  // NOLINT

void SplitCommandlineArg(const std::string& commandline,
                         int* argc,
                         char*** argv) {
  static std::vector<std::string> args;
  args.clear();
  for (size_t start_pos = 0, end_pos = 0;
       start_pos < commandline.size() && end_pos != std::string::npos;
       start_pos = end_pos + 1) {
    end_pos = commandline.find(' ', start_pos);
    args.push_back(commandline.substr(start_pos, end_pos - start_pos));
  }
  args.push_back("");  // test empty argument
  *argc = static_cast<int>(args.size());
  *argv = new char*[*argc];
  for (size_t i = 0; i < args.size(); i++) {
    (*argv)[i] = const_cast<char*>(args[i].c_str());
  }
}

TEST(flags_native_test, ParseCommandLineFlags) {
  uint32_t test_int32 = 2;
  ASSERT_EQ(FLAGS_paddle_test_int32, 1);
  ASSERT_EQ(FLAGS_paddle_test_uint32, test_int32);
  ASSERT_EQ(FLAGS_paddle_test_string, "raw");

  // Construct commandline arguments input
  std::string commandline =
      "test --paddle_test_int32=3 --paddle_test_uint32=\"4\" "
      "--paddle_test_string \"modified string\"";
  int argc = 0;
  char** argv = nullptr;
  SplitCommandlineArg(commandline, &argc, &argv);

  // Parse commandline flags and check
  ParseCommandLineFlags(&argc, &argv);
  delete argv;

  test_int32 = 4;
  ASSERT_EQ(FLAGS_paddle_test_int32, 3);
  ASSERT_EQ(FLAGS_paddle_test_uint32, test_int32);
  ASSERT_EQ(FLAGS_paddle_test_string, "modified string");

  // test FindFlag and SetFlagValue
  ASSERT_TRUE(FindFlag("paddle_test_int32"));

  SetFlagValue("paddle_test_int32", "9");
  ASSERT_EQ(FLAGS_paddle_test_int32, 9);
}

#if defined(_POSIX_C_SOURCE) && \
    _POSIX_C_SOURCE >= 200112L  // environment for use setenv
bool SetEnvVar(const std::string& var_name, const std::string& var_value) {
  int res = setenv(var_name.c_str(), var_value.c_str(), 1);
  return res == 0;
}

PD_DEFINE_bool(paddle_test_env_bool, false, "test env bool flag");
PD_DEFINE_double(paddle_test_env_double, 3.14, "test env double flag");

TEST(flags_native_test, SetFlagsFromEnv) {
  ASSERT_EQ(FLAGS_paddle_test_env_bool, false);
  ASSERT_EQ(FLAGS_paddle_test_env_double, 3.14);

  ASSERT_TRUE(SetEnvVar("FLAGS_paddle_test_env_bool", "true"));
  ASSERT_TRUE(SetEnvVar("FLAGS_paddle_test_env_double", "2.71"));

  // test GetFromEnv
  ASSERT_EQ(GetFromEnv<bool>("FLAGS_paddle_test_env_bool", false), true);
  ASSERT_EQ(GetFromEnv<int32_t>("FLAGS_int32_not_defined", 34), 34);

  // test SetFlagsFromEnv
  std::string commandline =
      "test --fromenv=paddle_test_env_bool,paddle_test_env_double";
  int argc;
  char** argv;
  SplitCommandlineArg(commandline, &argc, &argv);
  ParseCommandLineFlags(&argc, &argv);
  delete argv;

  ASSERT_EQ(FLAGS_paddle_test_env_bool, true);
  ASSERT_EQ(FLAGS_paddle_test_env_double, 2.71);
}
#endif

TEST(flags_native_test, PrintAllFlagHelp) { PrintAllFlagHelp(); }
