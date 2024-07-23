/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/common/errors.h"

#include <string>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/enforce.h"

using namespace phi::errors;  // NOLINT

#define CHECK_PADDLE_THROW(EFUNC)                                          \
  do {                                                                     \
    bool caught_exception = false;                                         \
    try {                                                                  \
      PADDLE_THROW((EFUNC)("paddle throw test."));                         \
    } catch (paddle::platform::EnforceNotMet & error) {                    \
      caught_exception = true;                                             \
      std::string ex_msg = error.what();                                   \
      EXPECT_TRUE(ex_msg.find("paddle throw test.") != std::string::npos); \
    }                                                                      \
    EXPECT_TRUE(caught_exception);                                         \
  } while (0)

#define CHECK_PADDLE_ENFORCE(EFUNC)                                          \
  do {                                                                       \
    bool caught_exception = false;                                           \
    try {                                                                    \
      PADDLE_ENFORCE(false, (EFUNC)("paddle enforce test."));                \
    } catch (paddle::platform::EnforceNotMet & error) {                      \
      caught_exception = true;                                               \
      std::string ex_msg = error.what();                                     \
      EXPECT_TRUE(ex_msg.find("paddle enforce test.") != std::string::npos); \
    }                                                                        \
    EXPECT_TRUE(caught_exception);                                           \
  } while (0)

#define CHECK_PADDLE_ENFORCE_NOT_NULL(EFUNC)                             \
  do {                                                                   \
    bool caught_exception = false;                                       \
    try {                                                                \
      PADDLE_ENFORCE_NOT_NULL(nullptr,                                   \
                              (EFUNC)("paddle enforce not null test.")); \
    } catch (paddle::platform::EnforceNotMet & error) {                  \
      caught_exception = true;                                           \
      std::string ex_msg = error.what();                                 \
      EXPECT_TRUE(ex_msg.find("paddle enforce not null test.") !=        \
                  std::string::npos);                                    \
    }                                                                    \
    EXPECT_TRUE(caught_exception);                                       \
  } while (0)

#define CHECK_PADDLE_ENFORCE_EQ(EFUNC)                                \
  do {                                                                \
    bool caught_exception = false;                                    \
    try {                                                             \
      PADDLE_ENFORCE_EQ(1, 2, (EFUNC)("paddle enforce equal test.")); \
    } catch (paddle::platform::EnforceNotMet & error) {               \
      caught_exception = true;                                        \
      std::string ex_msg = error.what();                              \
      EXPECT_TRUE(ex_msg.find("paddle enforce equal test.") !=        \
                  std::string::npos);                                 \
    }                                                                 \
    EXPECT_TRUE(caught_exception);                                    \
  } while (0)

#define CHECK_ALL_PADDLE_EXCEPTION_MACRO(EFUNC) \
  do {                                          \
    CHECK_PADDLE_THROW(EFUNC);                  \
    CHECK_PADDLE_ENFORCE(EFUNC);                \
    CHECK_PADDLE_ENFORCE_NOT_NULL(EFUNC);       \
    CHECK_PADDLE_ENFORCE_EQ(EFUNC);             \
  } while (0)

TEST(Errors, InvalidArgument) {
  CHECK_ALL_PADDLE_EXCEPTION_MACRO(InvalidArgument);
}

TEST(Errors, NotFound) { CHECK_ALL_PADDLE_EXCEPTION_MACRO(NotFound); }

TEST(Errors, OutOfRange) { CHECK_ALL_PADDLE_EXCEPTION_MACRO(OutOfRange); }

TEST(Errors, AlreadExists) { CHECK_ALL_PADDLE_EXCEPTION_MACRO(AlreadyExists); }

TEST(Errors, ResourceExhausted) {
  CHECK_ALL_PADDLE_EXCEPTION_MACRO(ResourceExhausted);
}

TEST(Errors, PreconditionNotMet) {
  CHECK_ALL_PADDLE_EXCEPTION_MACRO(PreconditionNotMet);
}

TEST(Errors, PermissionDenied) {
  CHECK_ALL_PADDLE_EXCEPTION_MACRO(PermissionDenied);
}

TEST(Errors, ExecutionTimeout) {
  CHECK_ALL_PADDLE_EXCEPTION_MACRO(ExecutionTimeout);
}

TEST(Errors, Unimplemented) { CHECK_ALL_PADDLE_EXCEPTION_MACRO(Unimplemented); }

TEST(Errors, Unavailable) { CHECK_ALL_PADDLE_EXCEPTION_MACRO(Unavailable); }

TEST(Errors, Fatal) { CHECK_ALL_PADDLE_EXCEPTION_MACRO(Fatal); }

TEST(Errors, External) { CHECK_ALL_PADDLE_EXCEPTION_MACRO(External); }
