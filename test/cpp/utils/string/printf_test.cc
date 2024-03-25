//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/utils/string/printf.h"

#include <string>

#include "gtest/gtest.h"

TEST(StringPrintf, StringPrintf) {
  std::string weekday = "Wednesday";
  const char* month = "July";
  size_t day = 27;
  int hour = 14;
  int min = 44;
  EXPECT_EQ(std::string("Wednesday, July 27, 14:44"),
            paddle::string::Sprintf(
                "%s, %s %d, %.2d:%.2d", weekday, month, day, hour, min));
  EXPECT_EQ(std::string(""), paddle::string::Sprintf());
}
