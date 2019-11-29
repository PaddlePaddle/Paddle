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
#include <sstream>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/ddim.h"

TEST(DDim, Equality) {
  // construct a DDim from an initialization list
  paddle::framework::DDim ddim = paddle::framework::make_ddim({9, 1, 5});
  EXPECT_EQ(ddim[0], 9);
  EXPECT_EQ(ddim[1], 1);
  EXPECT_EQ(ddim[2], 5);

  // construct a DDim from a vector
  std::vector<int64_t> vec({9, 1, 5});
  paddle::framework::DDim vddim = paddle::framework::make_ddim(vec);
  EXPECT_EQ(ddim[0], 9);
  EXPECT_EQ(ddim[1], 1);
  EXPECT_EQ(ddim[2], 5);

  // mutate a DDim
  ddim[1] = 2;
  EXPECT_EQ(ddim[1], 2);
  ddim[0] = 6;
  EXPECT_EQ(ddim[0], 6);

  // vectorize a DDim
  std::vector<int64_t> res_vec = paddle::framework::vectorize(vddim);
  EXPECT_EQ(res_vec[0], 9);
  EXPECT_EQ(res_vec[1], 1);
  EXPECT_EQ(res_vec[2], 5);
  paddle::framework::Dim<3> d(3, 2, 1);
  res_vec = paddle::framework::vectorize(paddle::framework::DDim(d));
  EXPECT_EQ(res_vec[0], 3);
  EXPECT_EQ(res_vec[1], 2);
  EXPECT_EQ(res_vec[2], 1);

  // arity of a DDim
  EXPECT_EQ(paddle::framework::arity(ddim), 3);
  EXPECT_EQ(ddim.size(), 3);

  // product of a DDim
  EXPECT_EQ(paddle::framework::product(vddim), 45);
  EXPECT_EQ(
      paddle::framework::product(paddle::framework::make_ddim({3, 2, 5, 3})),
      90);

  // slice a DDim
  paddle::framework::DDim ddim2 =
      paddle::framework::make_ddim({1, 2, 3, 4, 5, 6});
  paddle::framework::DDim ss = paddle::framework::slice_ddim(ddim2, 2, 5);
  EXPECT_EQ(arity(ss), 3);
  EXPECT_EQ(ss[0], 3);
  EXPECT_EQ(ss[1], 4);
  EXPECT_EQ(ss[2], 5);
  paddle::framework::DDim ss2 = paddle::framework::slice_ddim(ddim2, 0, 6);
  EXPECT_EQ(arity(ss2), 6);
  EXPECT_EQ(ss2[0], 1);
  EXPECT_EQ(ss2[1], 2);
  EXPECT_EQ(ss2[2], 3);
  EXPECT_EQ(ss2[3], 4);
  EXPECT_EQ(ss2[4], 5);
  EXPECT_EQ(ss2[5], 6);
}

TEST(DDim, Print) {
  // print a DDim
  std::stringstream ss;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({2, 3, 4});
  ss << ddim;
  EXPECT_EQ("2, 3, 4", ss.str());
}
