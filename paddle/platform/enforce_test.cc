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

#include "paddle/platform/enforce.h"
#include "gtest/gtest.h"

TEST(ENFORCE, OK) {
  PADDLE_ENFORCE(true, "Enforce is ok %d now %f", 123, 0.345);
  size_t val = 1;
  const size_t limit = 10;
  PADDLE_ENFORCE(val < limit, "Enforce is OK too");
}

TEST(ENFORCE, FAILED) {
  bool in_catch = false;
  try {
    PADDLE_ENFORCE(false, "Enforce is not ok %d at all", 123);
  } catch (const std::runtime_error& error) {
    // your error handling code here
    in_catch = true;
    std::string msg = "Enforce is not ok 123 at all";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }
  ASSERT_TRUE(in_catch);
}
