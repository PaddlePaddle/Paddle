/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/framework/tuple.h"

#include "gtest/gtest.h"

TEST(Tuple, Make) {
  std::vector<paddle::framework::ElementVar> element_type;
  element_type.push_back(12);
  element_type.push_back(12.0f);
  element_type.push_back("ElementVar");

  paddle::framework::Tuple* tuple = paddle::framework::make_tuple(element_type);

  EXPECT_EQ(PADDLE_GET(int, tuple->get(0)), 12);
  EXPECT_EQ(PADDLE_GET(float, tuple->get(1)), 12.0f);
  EXPECT_EQ(PADDLE_GET(std::string, tuple->get(2)), "ElementVar");

  delete tuple;
}

TEST(Tuple, IsTheSameType) {
  std::vector<paddle::framework::ElementVar> element_type1;
  std::vector<paddle::framework::ElementVar> element_type2;
  std::vector<paddle::framework::ElementVar> element_type3;

  element_type1.push_back(12);
  element_type1.push_back(12.0f);
  element_type1.push_back("Tuple1");

  element_type2.push_back(13);
  element_type2.push_back(13.0f);
  element_type2.push_back("Tuple2");

  element_type3.push_back(14.0f);
  element_type3.push_back(14);
  element_type3.push_back("Tuple3");

  paddle::framework::Tuple* tuple1 =
      paddle::framework::make_tuple(element_type1);
  paddle::framework::Tuple* tuple2 =
      paddle::framework::make_tuple(element_type2);
  paddle::framework::Tuple* tuple3 =
      paddle::framework::make_tuple(element_type3);

  EXPECT_TRUE(tuple1->isSameType(*tuple2));
  EXPECT_FALSE(tuple1->isSameType(*tuple3));

  delete tuple1;
  delete tuple2;
  delete tuple3;
}
