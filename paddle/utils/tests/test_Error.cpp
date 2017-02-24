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

#include "paddle/utils/Error.h"

#include <gtest/gtest.h>

TEST(Error, testAll) {
  paddle::Error error;
  ASSERT_FALSE(error);
  error = paddle::Error("I'm the error");
  ASSERT_TRUE(error);
  ASSERT_STREQ("I'm the error", error.msg());

  error = paddle::Error("error2");
  ASSERT_TRUE(error);
  ASSERT_STREQ("error2", error.msg());

  int i = 3;
  auto error3 = paddle::Error("error%d", i);
  ASSERT_TRUE(error3);
  ASSERT_STREQ("error3", error3.msg());
}
