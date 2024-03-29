// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <sstream>

#include "gtest/gtest.h"
#include "paddle/phi/core/ddim.h"

namespace phi {
namespace tests {

TEST(DDim, Equality) {
  // default construct ddim
  phi::DDim default_ddim;
  EXPECT_EQ(arity(default_ddim), -1);
  EXPECT_EQ(default_ddim[0], 0);

  // construct a zero-DDim
  phi::DDim zero_ddim = common::make_ddim({});
  EXPECT_EQ(arity(zero_ddim), 0);
  EXPECT_EQ(zero_ddim.size(), 0);
  EXPECT_EQ(common::product(zero_ddim), 1);

  std::vector<int64_t> zero_vec;
  phi::DDim zero_ddim1 = common::make_ddim(zero_vec);
  EXPECT_EQ(arity(zero_ddim1), 0);
  EXPECT_EQ(zero_ddim1.size(), 0);
  EXPECT_EQ(common::product(zero_ddim1), 1);

  // zero-DDim to vector
  std::vector<int64_t> zero_ddim_vec = common::vectorize(zero_ddim);
  EXPECT_EQ(zero_ddim_vec.size(), size_t(0));

  // reshape zero-DDim
  std::vector<int> reshape_vec = {1};
  phi::DDim reshape_ddim = zero_ddim.reshape(reshape_vec);
  EXPECT_EQ(arity(reshape_ddim), 1);
  EXPECT_EQ(reshape_ddim.size(), 1);
  EXPECT_EQ(common::product(reshape_ddim), 1);

  // construct a DDim from an initialization list
  phi::DDim ddim = common::make_ddim({9, 1, 5});
  EXPECT_EQ(ddim[0], 9);
  EXPECT_EQ(ddim[1], 1);
  EXPECT_EQ(ddim[2], 5);

  // arity of a DDim
  EXPECT_EQ(common::arity(ddim), 3);
  EXPECT_EQ(ddim.size(), 3);

  // mutate a DDim
  ddim[1] = 2;
  EXPECT_EQ(ddim[1], 2);
  ddim[0] = 6;
  EXPECT_EQ(ddim[0], 6);

  // construct a DDim from a vector
  std::vector<int64_t> vec({9, 1, 5});
  phi::DDim vddim = common::make_ddim(vec);
  EXPECT_EQ(vddim[0], 9);
  EXPECT_EQ(vddim[1], 1);
  EXPECT_EQ(vddim[2], 5);

  // vectorize a DDim
  std::vector<int64_t> res_vec = common::vectorize(vddim);
  EXPECT_EQ(res_vec[0], 9);
  EXPECT_EQ(res_vec[1], 1);
  EXPECT_EQ(res_vec[2], 5);
  phi::Dim<3> d(3, 2, 1);
  res_vec = common::vectorize(phi::DDim(d));
  EXPECT_EQ(res_vec[0], 3);
  EXPECT_EQ(res_vec[1], 2);
  EXPECT_EQ(res_vec[2], 1);

  // product of a DDim
  EXPECT_EQ(common::product(vddim), 45);
  EXPECT_EQ(common::product(common::make_ddim({3, 2, 5, 3})), 90);

  // slice a DDim
  phi::DDim ddim2 = common::make_ddim({1, 2, 3, 4, 5, 6});
  phi::DDim slice_dim1 = common::slice_ddim(ddim2, 2, 5);
  EXPECT_EQ(arity(slice_dim1), 3);
  EXPECT_EQ(slice_dim1[0], 3);
  EXPECT_EQ(slice_dim1[1], 4);
  EXPECT_EQ(slice_dim1[2], 5);

  phi::DDim slice_dim2 = common::slice_ddim(ddim2, 0, 6);
  EXPECT_EQ(arity(slice_dim2), 6);
  EXPECT_EQ(slice_dim2[0], 1);
  EXPECT_EQ(slice_dim2[1], 2);
  EXPECT_EQ(slice_dim2[2], 3);
  EXPECT_EQ(slice_dim2[3], 4);
  EXPECT_EQ(slice_dim2[4], 5);
  EXPECT_EQ(slice_dim2[5], 6);

  phi::DDim slice_dim3 = common::slice_ddim(ddim2, 1, 1);
  EXPECT_EQ(arity(slice_dim3), 0);
  EXPECT_EQ(slice_dim3.size(), 0);
  EXPECT_EQ(common::product(slice_dim3), 1);
}

TEST(DDim, Print) {
  // print a DDim
  std::stringstream ss1;
  phi::DDim ddim = common::make_ddim({2, 3, 4});
  ss1 << ddim;
  EXPECT_EQ("2, 3, 4", ss1.str());

  // print a zero-DDim
  std::stringstream ss2;
  phi::DDim zero_ddim = common::make_ddim({});
  ss2 << zero_ddim;
  EXPECT_EQ("", ss2.str());
}

TEST(DDim, Hash) {
  // hash a DDim
  std::size_t h = 0;
  phi::DDim ddim = common::make_ddim({2, 3, 4});
  h = std::hash<phi::DDim>()(ddim);
  EXPECT_EQ(h, 0xa16fb2b2967ul);
}

}  // namespace tests
}  // namespace phi
