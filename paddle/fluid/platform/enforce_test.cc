/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <array>
#include <iostream>
#include <memory>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/piece.h"

using StringPiece = paddle::string::Piece;
using paddle::string::HasPrefix;

TEST(ENFORCE, OK) {
  PADDLE_ENFORCE(true, "Enforce is ok %d now %f", 123, 0.345);
  size_t val = 1;
  const size_t limit = 10;
  PADDLE_ENFORCE(val < limit, "Enforce is OK too");
}

TEST(ENFORCE, FAILED) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE(false, "Enforce is not ok %d at all", 123);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_TRUE(
        HasPrefix(StringPiece(error.what()), "Enforce is not ok 123 at all"));
  }
  EXPECT_TRUE(caught_exception);

  caught_exception = false;
  try {
    PADDLE_ENFORCE(false, "Enforce is not ok at all");
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_TRUE(
        HasPrefix(StringPiece(error.what()), "Enforce is not ok at all"));
  }
  EXPECT_TRUE(caught_exception);

  caught_exception = false;
  try {
    PADDLE_ENFORCE(false);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_NE(std::string(error.what()).find("  at "), 0);
  }
  EXPECT_TRUE(caught_exception);
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
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_EQ(a, 1 + 3);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    HasPrefix(
        StringPiece(error.what()),
        "Enforce failed. Expected a == 1 + 3, but received a:2 != 1 + 3:4.");
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_EQ, EXTRA_MSG_FAIL) {
  int a = 2;
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_EQ(a, 1 + 3, "%s size not match", "their");
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    HasPrefix(StringPiece(error.what()),
              "Enforce failed. Expected a == 1 + 3, but received a:2 != 1 + "
              "3:4.\ntheir size not match");
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_NE, OK) {
  PADDLE_ENFORCE_NE(1, 2);
  PADDLE_ENFORCE_NE(1.0, 2UL);
}
TEST(ENFORCE_NE, FAIL) {
  bool caught_exception = false;

  try {
    // 2UL here to check data type compatible
    PADDLE_ENFORCE_NE(1.0, 1UL);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_TRUE(HasPrefix(
        StringPiece(error.what()),
        "Enforce failed. Expected 1.0 != 1UL, but received 1.0:1 == 1UL:1."))
        << error.what() << " does not have expected prefix";
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_GT, OK) { PADDLE_ENFORCE_GT(2, 1); }
TEST(ENFORCE_GT, FAIL) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_GT(1, 2);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_TRUE(
        HasPrefix(StringPiece(error.what()),
                  "Enforce failed. Expected 1 > 2, but received 1:1 <= 2:2."));
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_GE, OK) {
  PADDLE_ENFORCE_GE(2, 2);
  PADDLE_ENFORCE_GE(3, 2);
  PADDLE_ENFORCE_GE(3.21, 2.0);
}
TEST(ENFORCE_GE, FAIL) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_GE(1, 2);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_TRUE(
        HasPrefix(StringPiece(error.what()),
                  "Enforce failed. Expected 1 >= 2, but received 1:1 < 2:2."));
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_LE, OK) {
  PADDLE_ENFORCE_LE(1, 1);
  PADDLE_ENFORCE_LE(1UL, 1UL);
  PADDLE_ENFORCE_LE(2, 3);
  PADDLE_ENFORCE_LE(2UL, 3UL);
  PADDLE_ENFORCE_LE(2.0, 3.2);
}
TEST(ENFORCE_LE, FAIL) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_GT(1, 2);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_TRUE(
        HasPrefix(StringPiece(error.what()),
                  "Enforce failed. Expected 1 > 2, but received 1:1 <= 2:2."));
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_LT, OK) {
  PADDLE_ENFORCE_LT(3, 10);
  PADDLE_ENFORCE_LT(2UL, 3UL);
  PADDLE_ENFORCE_LT(2, 3);
}
TEST(ENFORCE_LT, FAIL) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_LT(1UL, 0.12);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_TRUE(HasPrefix(StringPiece(error.what()),
                          "Enforce failed. Expected 1UL < 0.12, but "
                          "received 1UL:1 >= 0.12:0.12."));
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_NOT_NULL, OK) {
  int* a = new int;
  PADDLE_ENFORCE_NOT_NULL(a);
  delete a;
}
TEST(ENFORCE_NOT_NULL, FAIL) {
  bool caught_exception = false;
  try {
    int* a = nullptr;
    PADDLE_ENFORCE_NOT_NULL(a);
  } catch (paddle::platform::EnforceNotMet error) {
    caught_exception = true;
    EXPECT_TRUE(HasPrefix(StringPiece(error.what()), "a should not be null"));
  }
  EXPECT_TRUE(caught_exception);
}

struct Dims {
  size_t dims_[4];

  bool operator==(const Dims& o) const {
    for (size_t i = 0; i < 4; ++i) {
      if (dims_[i] != o.dims_[i]) return false;
    }
    return true;
  }
};

std::ostream& operator<<(std::ostream& os, const Dims& d) {
  for (size_t i = 0; i < 4; ++i) {
    if (i == 0) {
      os << "[";
    }
    os << d.dims_[i];
    if (i == 4 - 1) {
      os << "]";
    } else {
      os << ", ";
    }
  }
  return os;
}

TEST(ENFORCE_USER_DEFINED_CLASS, EQ) {
  Dims a{{1, 2, 3, 4}}, b{{1, 2, 3, 4}};
  PADDLE_ENFORCE_EQ(a, b);
}

TEST(ENFORCE_USER_DEFINED_CLASS, NE) {
  Dims a{{1, 2, 3, 4}}, b{{5, 6, 7, 8}};
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_EQ(a, b);
  } catch (paddle::platform::EnforceNotMet&) {
    caught_exception = true;
  }
  EXPECT_TRUE(caught_exception);
}

TEST(EOF_EXCEPTION, THROW_EOF) {
  bool caught_eof = false;
  try {
    PADDLE_THROW_EOF();
  } catch (paddle::platform::EOFException error) {
    caught_eof = true;
    EXPECT_TRUE(HasPrefix(StringPiece(error.what()), "There is no next data."));
  }
  EXPECT_TRUE(caught_eof);
}
