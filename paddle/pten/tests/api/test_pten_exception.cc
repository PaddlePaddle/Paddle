/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include <string>
#include "gtest/gtest.h"
#include "paddle/pten/api/ext/exception.h"

TEST(PD_THROW, empty) {
  bool caught_exception = false;
  try {
    PD_THROW();
  } catch (const std::exception& e) {
    caught_exception = true;
    std::string err_msg = e.what();
    EXPECT_TRUE(err_msg.find("An error occurred.") != std::string::npos);
#if _WIN32
    EXPECT_TRUE(err_msg.find("tests\\api\\test_pten_exception.cc:20") !=
                std::string::npos);
#else
    EXPECT_TRUE(
        err_msg.find("paddle/pten/tests/api/test_pten_exception.cc:20") !=
        std::string::npos);
#endif
  }
  EXPECT_TRUE(caught_exception);
}

TEST(PD_THROW, non_empty) {
  bool caught_exception = false;
  try {
    PD_THROW("PD_THROW returns ",
             false,
             ". DataType of ",
             1,
             " is INT. ",
             "DataType of ",
             0.23,
             " is FLOAT. ");
  } catch (const std::exception& e) {
    caught_exception = true;
    std::string err_msg = e.what();
    EXPECT_TRUE(err_msg.find("PD_THROW returns 0. DataType of 1 is INT. ") !=
                std::string::npos);
#if _WIN32
    EXPECT_TRUE(err_msg.find("tests\\api\\test_pten_exception.cc") !=
                std::string::npos);
#else
    EXPECT_TRUE(err_msg.find("paddle/pten/tests/api/test_pten_exception.cc") !=
                std::string::npos);
#endif
  }
  EXPECT_TRUE(caught_exception);
}

TEST(PD_CHECK, OK) {
  PD_CHECK(true);
  PD_CHECK(true, "PD_CHECK returns ", true, "now");

  const size_t a = 1;
  const size_t b = 10;
  PD_CHECK(a < b);
  PD_CHECK(a < b, "PD_CHECK returns ", true, a, "should < ", b);
}

TEST(PD_CHECK, FAILED) {
  bool caught_exception = false;
  try {
    PD_CHECK(false);
  } catch (const std::exception& e) {
    caught_exception = true;
    std::string err_msg = e.what();
    EXPECT_TRUE(err_msg.find("Expected false, but it's not satisfied.") !=
                std::string::npos);
#if _WIN32
    EXPECT_TRUE(err_msg.find("tests\\api\\test_pten_exception.cc") !=
                std::string::npos);
#else
    EXPECT_TRUE(err_msg.find("paddle/pten/tests/api/test_pten_exception.cc") !=
                std::string::npos);
#endif
  }
  EXPECT_TRUE(caught_exception);

  caught_exception = false;
  try {
    PD_CHECK(false,
             "PD_CHECK returns ",
             false,
             ". DataType of ",
             1,
             " is INT. ",
             "DataType of ",
             0.23,
             " is FLOAT. ");
  } catch (const std::exception& e) {
    caught_exception = true;
    std::string err_msg = e.what();
    EXPECT_TRUE(err_msg.find("PD_CHECK returns 0. DataType of 1 is INT. ") !=
                std::string::npos);
#if _WIN32
    EXPECT_TRUE(err_msg.find("tests\\api\\test_pten_exception.cc") !=
                std::string::npos);
#else
    EXPECT_TRUE(err_msg.find("paddle/pten/tests/api/test_pten_exception.cc") !=
                std::string::npos);
#endif
  }
  EXPECT_TRUE(caught_exception);

  const size_t a = 1;
  const size_t b = 10;
  caught_exception = false;
  try {
    PD_CHECK(a > b);
  } catch (const std::exception& e) {
    caught_exception = true;
    std::string err_msg = e.what();
    EXPECT_TRUE(err_msg.find("Expected a > b, but it's not satisfied.") !=
                std::string::npos);
#if _WIN32
    EXPECT_TRUE(err_msg.find("tests\\api\\test_pten_exception.cc") !=
                std::string::npos);
#else
    EXPECT_TRUE(err_msg.find("paddle/pten/tests/api/test_pten_exception.cc") !=
                std::string::npos);
#endif
  }
  EXPECT_TRUE(caught_exception);

  const size_t c = 123;
  const float d = 0.345;
  caught_exception = false;
  try {
    PD_CHECK(c < d, "PD_CHECK returns ", false, ", because ", c, " > ", d);
  } catch (const std::exception& e) {
    caught_exception = true;
    std::string err_msg = e.what();
    EXPECT_TRUE(err_msg.find("PD_CHECK returns 0, because 123 > 0.345") !=
                std::string::npos);
#if _WIN32
    EXPECT_TRUE(err_msg.find("tests\\api\\test_pten_exception.cc") !=
                std::string::npos);
#else
    EXPECT_TRUE(err_msg.find("paddle/pten/tests/api/test_pten_exception.cc") !=
                std::string::npos);
#endif
  }
  EXPECT_TRUE(caught_exception);
}
