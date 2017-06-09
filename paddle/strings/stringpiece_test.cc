/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "paddle/strings/stringpiece.h"
#include "gtest/gtest.h"

TEST(StringPiece, Construct) {
  {
    paddle::StringPiece s;
    EXPECT_EQ(NULL, s.data());
    EXPECT_EQ(0U, s.len());
    EXPECT_EQ(0U, s.cap());
  }
  {
    EXPECT_THROW([] { paddle::StringPiece s(NULL, 10000U); }(),
                 std::invalid_argument);
  }
  {
    paddle::StringPiece s(NULL);
    EXPECT_EQ(0U, s.len());
  }
  {
    std::string a;
    EXPECT_EQ(0U, a.size());
    paddle::StringPiece s(a);
    EXPECT_EQ(0U, s.len());
  }
}

TEST(StringPiece, CopyAndAssign) {
  paddle::StringPiece empty;
  EXPECT_EQ(0U, empty.len());
  EXPECT_EQ(0U, empty.cap());

  paddle::StringPiece a("hello");
  paddle::StringPiece b = a;
  EXPECT_EQ(b.len(), strlen("hello"));
  EXPECT_EQ(b.cap(), strlen("hello"));
  EXPECT_EQ(a, b);

  std::string storage("hello");
  paddle::StringPiece c(storage);
  EXPECT_EQ(a, c);
  EXPECT_NE(a.data(), c.data());
}
