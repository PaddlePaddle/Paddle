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

#include "paddle/utils/Status.h"

#include <gtest/gtest.h>

TEST(Status, testAll) {
  paddle::Status status;
  ASSERT_TRUE(status.isOK());
  status.set("I'm the error");
  ASSERT_FALSE(status.isOK());
  ASSERT_STREQ("I'm the error", status.what());

  paddle::Status status2("error2");
  ASSERT_FALSE(status2.isOK());
  ASSERT_STREQ("error2", status2.what());

  int i = 3;
  auto status3 = paddle::Status::printf("error%d", i);
  ASSERT_FALSE(status3.isOK());
  ASSERT_STREQ("error3", status3.what());
}
