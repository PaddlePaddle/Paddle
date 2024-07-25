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

#include "paddle/utils/array_ref.h"

#include <array>
#include <cstdlib>
#include <ctime>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(array_ref, array_ref) {
  paddle::array_ref<int> a;
  CHECK_EQ(a.size(), size_t(0));
  CHECK_EQ(a.data(), static_cast<int*>(nullptr));

  paddle::array_ref<int> b(paddle::none);
  CHECK_EQ(b.size(), size_t(0));
  CHECK_EQ(b.data(), static_cast<int*>(nullptr));

  int v = 1;
  paddle::array_ref<int> c(v);
  CHECK_EQ(c.size(), size_t(1));
  CHECK_EQ(c.data(), &v);
  CHECK_EQ(c.equals(paddle::make_array_ref(v)), true);

  std::array<int, 5> v1 = {1, 2, 3, 4, 5};
  paddle::array_ref<int> d(v1.data(), 5);
  CHECK_EQ(d.size(), size_t(5));
  CHECK_EQ(d.data(), v1.data());
  CHECK_EQ(d.equals(paddle::make_array_ref(v1.data(), 5)), true);

  paddle::array_ref<int> e(&v1[0], &v1[4]);
  CHECK_EQ(e.size(), size_t(4));
  CHECK_EQ(e.data(), v1.data());
  CHECK_EQ(e.equals(paddle::make_array_ref(&v1[0], &v1[4])), true);

  paddle::small_vector<int, 3> small_vector{1, 2, 3};
  paddle::array_ref<int> f(small_vector);
  CHECK_EQ(f.size(), size_t(3));
  CHECK_EQ(f.data(), small_vector.data());
  CHECK_EQ(f.equals(paddle::make_array_ref(small_vector)), true);

  std::vector<int> vector{1, 2, 3};
  paddle::array_ref<int> g(vector);
  CHECK_EQ(g.size(), size_t(3));
  CHECK_EQ(g.data(), vector.data());
  CHECK_EQ(g.equals(paddle::make_array_ref(vector)), true);

  std::initializer_list<int> list = {1, 2, 3};
  paddle::array_ref<int> h(list);
  CHECK_EQ(h.size(), size_t(3));
  CHECK_EQ(h.data(), list.begin());

  paddle::array_ref<int> i(h);
  CHECK_EQ(i.size(), size_t(3));
  CHECK_EQ(i.data(), list.begin());
  CHECK_EQ(i.equals(h), true);
  CHECK_EQ(i.equals(paddle::make_array_ref(h)), true);

  auto slice = i.slice(1, 2);
  CHECK_EQ(slice.size(), size_t(2));
  CHECK_EQ(slice[0], 2);
  CHECK_EQ(slice[1], 3);

  auto drop = i.drop_front(2);
  CHECK_EQ(drop.size(), size_t(1));
  CHECK_EQ(drop[0], 3);

  static paddle::array_ref<int> nums = {1, 2, 3, 4, 5, 6, 7, 8};
  auto front = nums.take_front(3);
  CHECK_EQ(front.size(), size_t(3));
  for (size_t i = 0; i < 3; ++i) {
    CHECK_EQ(front[i], nums[i]);
  }
  auto back = nums.take_back(3);
  CHECK_EQ(back.size(), size_t(3));
  for (size_t i = 0; i < 3; ++i) {
    CHECK_EQ(back[i], nums[i + 5]);
  }
}
