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

#include <gtest/gtest.h>
#include <paddle/utils/ForceLink.h>
#include "test_ClassRegistrarLib.h"
// Enable link test_ClassRegistrarLib.cpp
PADDLE_ENABLE_FORCE_LINK_FILE(test_registrar);

TEST(ClassRegistrar, test) {
  std::vector<std::string> types;
  gTestRegistrar_.forEachType(
      [&types](const std::string& tp) { types.push_back(tp); });
  ASSERT_EQ(1, types.size());
  ASSERT_EQ("test", types[0]);
}
