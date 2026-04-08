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

#pragma once

#include <exception>
#include <string>
#include "gtest/gtest.h"

namespace test {
namespace utils {

template <typename ExceptionType = std::exception, typename Fn>
inline void ExpectThrowContains(Fn&& fn,
                                const std::string& expected_substr,
                                const std::string& context = "") {
  try {
    fn();
    if (context.empty()) {
      FAIL() << "Expected exception containing: " << expected_substr;
    } else {
      FAIL() << "Expected exception containing: " << expected_substr
             << ", context: " << context;
    }
  } catch (const ExceptionType& e) {
    if (context.empty()) {
      EXPECT_NE(std::string(e.what()).find(expected_substr), std::string::npos)
          << "actual error: " << e.what();
    } else {
      EXPECT_NE(std::string(e.what()).find(expected_substr), std::string::npos)
          << "context: " << context << ", actual error: " << e.what();
    }
  } catch (...) {
    if (context.empty()) {
      FAIL() << "Expected the specified exception type";
    } else {
      FAIL() << "Expected the specified exception type, context: " << context;
    }
  }
}

}  // namespace utils
}  // namespace test
