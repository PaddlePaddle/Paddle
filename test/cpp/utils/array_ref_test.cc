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
  PADDLE_ENFORCE_EQ(
      a.size(), size_t(0),
      platform::errors::InvalidArgument(
          "The input array a's size should be %d. But received %d ",
          size_t(0), a.size())
  );
  CHECK_EQ(a.data(), static_cast<int*>(nullptr));

  paddle::array_ref<int> b(paddle::none);
  PADDLE_ENFORCE_EQ(
      b.size(), size_t(0),
      platform::errors::InvalidArgument(
          "The input array b's size should be %d. But received %d ",
          size_t(0), b.size())
  );
  CHECK_EQ(b.data(), static_cast<int*>(nullptr));

  int v = 1;
  paddle::array_ref<int> c(v);
  PADDLE_ENFORCE_EQ(
      c.size(), size_t(1),
      platform::errors::InvalidArgument(
          "The input array c's size should be %d. But received %d ",
          size_t(1), c.size())
  );
  CHECK_EQ(c.data(), &v);
  CHECK_EQ(c.equals(paddle::make_array_ref(v)), true);

  std::array<int, 5> v1 = {1, 2, 3, 4, 5};
  paddle::array_ref<int> d(v1.data(), 5);
  PADDLE_ENFORCE_EQ(
      d.size(), size_t(5),
      platform::errors::InvalidArgument(
          "The input array d's size should be %d. But received %d ",
          size_t(5), d.size())
  );
  CHECK_EQ(d.data(), v1.data());
  CHECK_EQ(d.equals(paddle::make_array_ref(v1.data(), 5)), true);

  paddle::array_ref<int> e(&v1[0], &v1[4]);
  PADDLE_ENFORCE_EQ(
      e.size(), size_t(4),
      platform::errors::InvalidArgument(
          "The input array e's size should be %d. But received %d ",
          size_t(4), e.size())
  );
  CHECK_EQ(e.data(), v1.data());
  CHECK_EQ(e.equals(paddle::make_array_ref(&v1[0], &v1[4])), true);

  paddle::small_vector<int, 3> small_vector{1, 2, 3};
  paddle::array_ref<int> f(small_vector);
  PADDLE_ENFORCE_EQ(
      f.size(), size_t(3),
      platform::errors::InvalidArgument(
          "The input array f's size should be %d. But received %d ",
          size_t(3), f.size())
  );
  CHECK_EQ(f.data(), small_vector.data());
  CHECK_EQ(f.equals(paddle::make_array_ref(small_vector)), true);

  std::vector<int> vector{1, 2, 3};
  paddle::array_ref<int> g(vector);
  PADDLE_ENFORCE_EQ(
      g.size(), size_t(3),
      platform::errors::InvalidArgument(
          "The input array g's size should be %d. But received %d ",
          size_t(3), g.size())
  );
  CHECK_EQ(g.data(), vector.data());
  CHECK_EQ(g.equals(paddle::make_array_ref(vector)), true);

  std::initializer_list<int> list = {1, 2, 3};
  paddle::array_ref<int> h(list);
  PADDLE_ENFORCE_EQ(
      h.size(), size_t(3),
      platform::errors::InvalidArgument(
          "The input array h's size should be %d. But received %d ",
          size_t(3), h.size())
  );
  CHECK_EQ(h.data(), list.begin());

  paddle::array_ref<int> i(h);
  PADDLE_ENFORCE_EQ(
      i.size(), size_t(3),
      platform::errors::InvalidArgument(
          "The input array i's size should be %d. But received %d ",
          size_t(3), i.size())
  );
  CHECK_EQ(i.data(), list.begin());
  CHECK_EQ(i.equals(h), true);
  CHECK_EQ(i.equals(paddle::make_array_ref(h)), true);

  auto slice = i.slice(1, 2);
  PADDLE_ENFORCE_EQ(
      slice.size(), size_t(2),
      platform::errors::InvalidArgument(
          "The slice array's size should be %d. But received %d ",
          size_t(2), slice.size())
  );
  CHECK_EQ(slice[0], 2);
  CHECK_EQ(slice[1], 3);

  auto drop = i.drop_front(2);
  PADDLE_ENFORCE_EQ(
      drop.size(), size_t(1),
      platform::errors::InvalidArgument(
          "The drop array's size should be %d. But received %d ",
          size_t(1), drop.size())
  );
  CHECK_EQ(drop[0], 3);

  static paddle::array_ref<int> nums = {1, 2, 3, 4, 5, 6, 7, 8};
  auto front = nums.take_front(3);
  PADDLE_ENFORCE_EQ(
      front.size(), size_t(3),
      platform::errors::InvalidArgument(
          "The front array's size should be %d. But received %d ",
          size_t(3), front.size())
  );
  for (size_t i = 0; i < 3; ++i) {
    CHECK_EQ(front[i], nums[i]);
  }
  auto back = nums.take_back(3);
  PADDLE_ENFORCE_EQ(
      back.size(), size_t(3),
      platform::errors::InvalidArgument(
          "The back array's size should be %d. But received %d ",
          size_t(3), back.size())
  );
  for (size_t i = 0; i < 3; ++i) {
    CHECK_EQ(back[i], nums[i + 5]);
  }
}
