// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/exception.h"

#include <optional>
#include <string>

#include "gtest/gtest.h"

TEST(error_message_test, optional_with_value) {
  std::optional<int> value = 42;
  std::string msg = common::ErrorMessage("value=", value).to_string();
  EXPECT_EQ(msg, "value=42");
}

TEST(error_message_test, optional_without_value) {
  std::optional<int> value = std::nullopt;
  std::string msg = common::ErrorMessage("value=", value).to_string();
  EXPECT_EQ(msg, "value=nullopt");
}

TEST(pd_check_test, condition_true_no_throw) {
  EXPECT_NO_THROW(PD_CHECK(true, "should not throw"));
}

TEST(pd_check_test, condition_false_throws) {
  EXPECT_THROW(PD_CHECK(false, "bad value"), ::common::PD_Exception);
}

TEST(pd_check_test, exception_message_contains_custom_msg) {
  try {
    PD_CHECK(1 == 2, "expected 1==2 but got mismatch");
    FAIL() << "expected PD_Exception to be thrown";
  } catch (const ::common::PD_Exception& e) {
    EXPECT_NE(std::string(e.what()).find("expected 1==2 but got mismatch"),
              std::string::npos);
  }
}

TEST(pd_check_test, condition_false_with_optional_in_message) {
  std::optional<int> opt = std::nullopt;
  try {
    PD_CHECK(opt.has_value(), "opt is ", opt);
    FAIL() << "expected PD_Exception to be thrown";
  } catch (const ::common::PD_Exception& e) {
    EXPECT_NE(std::string(e.what()).find("opt is nullopt"), std::string::npos);
  }
}
