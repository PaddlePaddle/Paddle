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

#include <memory>

#include "gtest/gtest.h"
#include "paddle/platform/enforce.h"

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
  } catch (paddle::platform::EnforceNotMet error) {
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

TEST(ENFORCE, NO_ARG_OK) {
  int a = 2;
  int b = 2;
  PADDLE_ENFORCE_EQ(a, b);
  // test enforce with extra message.
  PADDLE_ENFORCE_EQ(a, b, "some thing wrong %s", "info");
}

TEST(ENFORCE_EQ, NO_EXTRA_MSG_FAIL) {
  int a = 2;
  bool in_catch = false;

  try {
    PADDLE_ENFORCE_EQ(a, 1 + 3);

  } catch (paddle::platform::EnforceNotMet error) {
    in_catch = true;
    const std::string msg = "enforce a == 1 + 3 failed, 2 != 4";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}

TEST(ENFORCE_EQ, EXTRA_MSG_FAIL) {
  int a = 2;
  bool in_catch = false;

  try {
    PADDLE_ENFORCE_EQ(a, 1 + 3, "%s size not match", "their");

  } catch (paddle::platform::EnforceNotMet error) {
    in_catch = true;
    const std::string msg =
        "enforce a == 1 + 3 failed, 2 != 4\ntheir size not match";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}

TEST(ENFORCE_NE, OK) {
  PADDLE_ENFORCE_NE(1, 2);
  PADDLE_ENFORCE_NE(1.0, 2UL);
}
TEST(ENFORCE_NE, FAIL) {
  bool in_catch = false;

  try {
    // 2UL here to check data type compatible
    PADDLE_ENFORCE_NE(1.0, 1UL);

  } catch (paddle::platform::EnforceNotMet error) {
    in_catch = true;
    const std::string msg = "enforce 1.0 != 1UL failed, 1.000000 == 1";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}

TEST(ENFORCE_GT, OK) { PADDLE_ENFORCE_GT(2, 1); }
TEST(ENFORCE_GT, FAIL) {
  bool in_catch = false;

  try {
    // 2UL here to check data type compatible
    PADDLE_ENFORCE_GT(1, 2UL);

  } catch (paddle::platform::EnforceNotMet error) {
    in_catch = true;
    const std::string msg = "enforce 1 > 2UL failed, 1 <= 2";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}

TEST(ENFORCE_GE, OK) {
  PADDLE_ENFORCE_GE(2, 2UL);
  PADDLE_ENFORCE_GE(3, 2UL);
  PADDLE_ENFORCE_GE(3, 2);
  PADDLE_ENFORCE_GE(3.21, 2UL);
}
TEST(ENFORCE_GE, FAIL) {
  bool in_catch = false;

  try {
    PADDLE_ENFORCE_GE(1, 2UL);

  } catch (paddle::platform::EnforceNotMet error) {
    in_catch = true;
    const std::string msg = "enforce 1 >= 2UL failed, 1 < 2";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}

TEST(ENFORCE_LE, OK) {
  PADDLE_ENFORCE_LE(1, 1);
  PADDLE_ENFORCE_LE(1, 1UL);
  PADDLE_ENFORCE_LE(2, 3UL);
  PADDLE_ENFORCE_LE(2UL, 3);
  PADDLE_ENFORCE_LE(2UL, 3.2);
}
TEST(ENFORCE_LE, FAIL) {
  bool in_catch = false;

  try {
    PADDLE_ENFORCE_GT(1, 2UL);

  } catch (paddle::platform::EnforceNotMet error) {
    in_catch = true;
    const std::string msg = "enforce 1 > 2UL failed, 1 <= 2";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}

TEST(ENFORCE_LT, OK) {
  PADDLE_ENFORCE_LT(3, 10);
  PADDLE_ENFORCE_LT(2, 3UL);
  PADDLE_ENFORCE_LT(2UL, 3);
}
TEST(ENFORCE_LT, FAIL) {
  bool in_catch = false;

  try {
    PADDLE_ENFORCE_LT(1UL, 0.12);

  } catch (paddle::platform::EnforceNotMet error) {
    in_catch = true;
    const std::string msg = "enforce 1UL < 0.12 failed, 1 >= 0.12";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}

TEST(ENFORCE_NOT_NULL, OK) {
  int* a = new int;
  PADDLE_ENFORCE_NOT_NULL(a);
  delete a;
}
TEST(ENFORCE_NOT_NULL, FAIL) {
  bool in_catch = false;
  int* a{nullptr};

  try {
    PADDLE_ENFORCE_NOT_NULL(a);

  } catch (paddle::platform::EnforceNotMet error) {
    in_catch = true;
    const std::string msg = "a should not be null";
    const char* what = error.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }

  ASSERT_TRUE(in_catch);
}
