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

#include "paddle/utils/StringUtil.h"

#include <gtest/gtest.h>

TEST(StringUtil, to) {
  ASSERT_NEAR(paddle::str::to<double>("12.45"), 12.45, 1e-5);
  ASSERT_DEATH(paddle::str::to<double>("12.45x23"), ".*");
  ASSERT_DEATH(paddle::str::to<int>(""), ".*");
}
