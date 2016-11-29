/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_USE_GFLAGS
//! Test Command Line Parser for paddle internal implement.

#include <paddle/utils/CommandLineParser.h>
#include <gtest/gtest.h>

P_DEFINE_int32(i1, 1, "test int flag 1");
P_DEFINE_int32(i2, 2, "test int flag 2");

P_DEFINE_string(str1, "1", "test str flag 1");
P_DEFINE_string(str2, "2", "test str flag 2");

P_DEFINE_bool(b1, true, "test bool flag 1");
P_DEFINE_bool(b2, false, "test bool flag 2");

P_DEFINE_double(d1, 0.1, "test double flag 1");
P_DEFINE_double(d2, -42.3, "test double flag 2");

P_DEFINE_int64(l1, 1, "test int64 flag 1");
P_DEFINE_int64(l2, 2, "test int64 flag 2");

P_DEFINE_uint64(ul1, 32, "test uint64 flag 1");
P_DEFINE_uint64(ul2, 33, "test uint64 flag 2");

constexpr double EPSILON = 1e-5;

#define cc(x) const_cast<char*>((x))

TEST(CommandLineParser, defaultValue) {
  char* argv[] = {cc("test_program"), cc("--unused_flag=134")};
  int argc = sizeof(argv) / sizeof(char*);

  paddle::ParseCommandLineFlags(&argc, argv);

  // Check Default Value
  ASSERT_EQ(argc, 2);
  ASSERT_EQ(FLAGS_i1, 1);
  ASSERT_EQ(FLAGS_i2, 2);
  ASSERT_EQ(FLAGS_str1, "1");
  ASSERT_EQ(FLAGS_str2, "2");
  ASSERT_EQ(FLAGS_b1, true);
  ASSERT_EQ(FLAGS_b2, false);
  ASSERT_NEAR(FLAGS_d1, 0.1, EPSILON);
  ASSERT_NEAR(FLAGS_d2, -42.3, EPSILON);
  ASSERT_EQ(FLAGS_i1, 1);
  ASSERT_EQ(FLAGS_i2, 2);
  ASSERT_EQ(FLAGS_ul1, 32UL);
  ASSERT_EQ(FLAGS_ul2, 33UL);
}

TEST(CommandLineParser, normal) {
  char* argv[] = {
      cc("test_program"), cc("--i2=32"),              cc("--str1=abc"),
      cc("--b2=1"),       cc("-b1=False"),            cc("--d2=.34"),
      cc("--d1=0"),       cc("--l1=-12345678901234"), cc("-ul2=3212")};
  int argc = sizeof(argv) / sizeof(char*);
  paddle::ParseCommandLineFlags(&argc, argv);
  ASSERT_EQ(argc, 1);
  ASSERT_EQ(FLAGS_i2, 32);
  ASSERT_EQ(FLAGS_str1, "abc");
  ASSERT_EQ(FLAGS_b2, true);
  ASSERT_EQ(FLAGS_b1, false);
  ASSERT_NEAR(FLAGS_d2, 0.34, EPSILON);
  ASSERT_NEAR(FLAGS_d1, 0.0, EPSILON);
  ASSERT_EQ(FLAGS_l1, -12345678901234);
  ASSERT_EQ(FLAGS_ul2, 3212UL);
}

TEST(CommandLineParser, printHelp) {
  char* argv[] = {cc("test_program"), cc("--help")};
  int argc = sizeof(argv) / sizeof(char*);

  // Will Print Usage
  ASSERT_DEATH(paddle::ParseCommandLineFlags(&argc, argv), ".*test_program.*");
}

TEST(CommandLineParser, parseError) {
  char* argv[] = {cc("test_program"), cc("--i1=abc")};

  int argc = sizeof(argv) / sizeof(char*);
  ASSERT_DEATH(
      paddle::ParseCommandLineFlags(&argc, argv),
      "Parse command flag i1 error! User input is --i1=abc.*test_program.*");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#else

int main(int argc, char** argv) {
  return 0;
}

#endif
