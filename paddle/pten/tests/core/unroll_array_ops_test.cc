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

#include "paddle/pten/core/utils/unroll_array_ops.h"

#include <gtest/gtest.h>
#include <array>

namespace pten {
namespace framework {

template <typename T>
bool CheckEquality(const T* p, size_t n, T val) {
  return std::all_of(p, p + n, [val](const T& v) { return v == val; });
}

template <int D1, int D2>
bool FillConstantTestMain() {
  static_assert(D1 >= D2, "");
  std::array<int, D1> arr;
  arr.fill(0);

  UnrollFillConstant<D2>::Run(arr.data(), 1);
  return CheckEquality(arr.data(), D2, 1) &&
         CheckEquality(arr.data() + D2, arr.size() - D2, 0);
}

TEST(unroll_ops, fill_constant) {
  EXPECT_TRUE((FillConstantTestMain<9, 0>()));
  EXPECT_TRUE((FillConstantTestMain<9, 1>()));
  EXPECT_TRUE((FillConstantTestMain<9, 4>()));
  EXPECT_TRUE((FillConstantTestMain<9, 9>()));
}

TEST(unroll_ops, assign) {
  const int a[] = {1, 2, 3, 4, 5};
  int b[] = {0, 0, 0, 0, 0};
  UnrollAssign<3>::Run(a, b);
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[1], 2);
  EXPECT_EQ(b[2], 3);
  EXPECT_EQ(b[3], 0);
  EXPECT_EQ(b[4], 0);
}

TEST(unroll_ops, var_args_assign) {
  int a[] = {0, 0, 0};
  UnrollVarArgsAssign<int>::Run(a, 1, 2);
  EXPECT_EQ(a[0], 1);
  EXPECT_EQ(a[1], 2);
  EXPECT_EQ(a[2], 0);
}

TEST(unroll_ops, compare) {
  int a[] = {1, 2, 3};
  int b[] = {1, 2, 4};
  EXPECT_TRUE(UnrollCompare<2>::Run(a, b));
  EXPECT_FALSE(UnrollCompare<3>::Run(a, b));

  b[0] = -1;
  EXPECT_TRUE(UnrollCompare<0>::Run(a, b));
  EXPECT_FALSE(UnrollCompare<1>::Run(a, b));
}

TEST(unroll_ops, product) {
  int a[] = {2, 3, 4};
  EXPECT_EQ(UnrollProduct<3>::Run(a), a[0] * a[1] * a[2]);
}

}  // namespace framework
}  // namespace pten
