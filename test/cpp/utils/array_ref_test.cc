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
#include "paddle/common/enforce.h"

TEST(array_ref, array_ref) {
  paddle::array_ref<int> a;
  PADDLE_ENFORCE_EQ(a.size(),
                    size_t(0),
                    common::errors::InvalidArgument(
                        "Array's size is invalid, expected %d but received %d.",
                        size_t(0),
                        a.size()));
  PADDLE_ENFORCE_EQ(a.data(),
                    static_cast<int*>(nullptr),
                    common::errors::InvalidArgument(
                        "Array's data is invalid, expected %d but received %d.",
                        static_cast<int*>(nullptr),
                        a.data()));

  paddle::array_ref<int> b(paddle::none);
  PADDLE_ENFORCE_EQ(b.size(),
                    size_t(0),
                    common::errors::InvalidArgument(
                        "Array's size is invalid, expected %d but received %d.",
                        size_t(0),
                        b.size()));
  PADDLE_ENFORCE_EQ(b.data(),
                    static_cast<int*>(nullptr),
                    common::errors::InvalidArgument(
                        "Array's data is invalid, expected %d but received %d.",
                        static_cast<int*>(nullptr),
                        b.data()));

  int v = 1;
  paddle::array_ref<int> c(v);
  PADDLE_ENFORCE_EQ(
      c.size(),
      size_t(1),
      common::errors::InvalidArgument(
          "Array's size is invalid, expected %d(size_t(1)) but received %d.",
          size_t(1),
          c.size()));
  PADDLE_ENFORCE_EQ(
      c.data(),
      &v,
      common::errors::InvalidArgument(
          "Array's data is invalid, expected %d(&v) but received %d.",
          &v,
          c.data()));
  PADDLE_ENFORCE_EQ(c.equals(paddle::make_array_ref(v)),
                    true,
                    common::errors::InvalidArgument(
                        "The output of paddle::make_array_ref(v) is wrong."));

  std::array<int, 5> v1 = {1, 2, 3, 4, 5};
  paddle::array_ref<int> d(v1.data(), 5);
  PADDLE_ENFORCE_EQ(
      d.size(),
      size_t(5),
      common::errors::InvalidArgument(
          "Array's size is invalid, expected %d(size_t(5)) but received %d.",
          size_t(5),
          d.size()));
  PADDLE_ENFORCE_EQ(d.data(),
                    v1.data(),
                    common::errors::InvalidArgument(
                        "Array's data is invalid, expected %d but received %d.",
                        v1.data(),
                        d.data()));
  PADDLE_ENFORCE_EQ(
      d.equals(paddle::make_array_ref(v1.data(), 5)),
      true,
      common::errors::InvalidArgument(
          "The output of paddle::make_array_ref(v1.data(), 5) is wrong."));

  paddle::array_ref<int> e(&v1[0], &v1[4]);
  PADDLE_ENFORCE_EQ(
      e.size(),
      size_t(4),
      common::errors::InvalidArgument(
          "Array's size is invalid, expected %d(size_t(4)) but received %d.",
          size_t(4),
          e.size()));
  PADDLE_ENFORCE_EQ(
      e.data(),
      v1.data(),
      common::errors::InvalidArgument(
          "Array's data is invalid, expected %d(v1.data()) but received %d.",
          v1.data(),
          e.data()));
  PADDLE_ENFORCE_EQ(
      e.equals(paddle::make_array_ref(&v1[0], &v1[4])),
      true,
      common::errors::InvalidArgument(
          "The output of paddle::make_array_ref(&v1[0], &v1[4]) is wrong."));

  paddle::small_vector<int, 3> small_vector{1, 2, 3};
  paddle::array_ref<int> f(small_vector);
  PADDLE_ENFORCE_EQ(
      f.size(),
      size_t(3),
      common::errors::InvalidArgument(
          "Array's size is invalid, expected %d(size_t(3)) but received %d.",
          size_t(3),
          f.size()));
  PADDLE_ENFORCE_EQ(f.data(),
                    small_vector.data(),
                    common::errors::InvalidArgument(
                        "Array's data is invalid, expected %d but received %d.",
                        small_vector.data(),
                        f.data()));
  PADDLE_ENFORCE_EQ(
      f.equals(paddle::make_array_ref(small_vector)),
      true,
      common::errors::InvalidArgument(
          "The output of paddle::make_array_ref(small_vector) is wrong."));

  std::vector<int> vector{1, 2, 3};
  paddle::array_ref<int> g(vector);
  PADDLE_ENFORCE_EQ(
      g.size(),
      size_t(3),
      common::errors::InvalidArgument(
          "Array's size is invalid, expected %d(size_t(3)) but received %d.",
          size_t(3),
          g.size()));
  PADDLE_ENFORCE_EQ(g.data(),
                    vector.data(),
                    common::errors::InvalidArgument(
                        "Array's data is invalid, expected %d but received %d.",
                        vector.data(),
                        g.data()));
  PADDLE_ENFORCE_EQ(
      g.equals(paddle::make_array_ref(vector)),
      true,
      common::errors::InvalidArgument(
          "The output of paddle::make_array_ref(vector) is wrong."));

  std::initializer_list<int> list = {1, 2, 3};
  paddle::array_ref<int> h(list);
  PADDLE_ENFORCE_EQ(
      h.size(),
      size_t(3),
      common::errors::InvalidArgument(
          "Array's size is invalid, expected %d(size_t(3)) but received %d.",
          size_t(3),
          h.size()));
  PADDLE_ENFORCE_EQ(h.data(),
                    list.begin(),
                    common::errors::InvalidArgument(
                        "Array's data is invalid, expected %d but received %d.",
                        list.begin(),
                        h.data()));

  paddle::array_ref<int> i(h);
  PADDLE_ENFORCE_EQ(
      i.size(),
      size_t(3),
      common::errors::InvalidArgument(
          "Array's size is invalid, expected %d(size_t(3)) but received %d.",
          size_t(3),
          i.size()));
  PADDLE_ENFORCE_EQ(i.data(),
                    list.begin(),
                    common::errors::InvalidArgument(
                        "Array's data is invalid, expected %d but received %d.",
                        list.begin(),
                        i.data()));
  PADDLE_ENFORCE_EQ(
      i.equals(h),
      true,
      common::errors::InvalidArgument("Array i(h) is not equal with h"));
  PADDLE_ENFORCE_EQ(i.equals(paddle::make_array_ref(h)),
                    true,
                    common::errors::InvalidArgument(
                        "i(h) is not equal with paddle::make_array_ref(h)"));

  auto slice = i.slice(1, 2);
  PADDLE_ENFORCE_EQ(
      slice.size(),
      size_t(2),
      common::errors::InvalidArgument(
          "Slice's size is invalid, expected %d(size_t(2)) but received %d.",
          size_t(2),
          slice.size()));
  PADDLE_ENFORCE_EQ(
      slice[0],
      2,
      common::errors::InvalidArgument(
          "slice[0]'s value is invalid, expected 2 but received %d.",
          slice[0]));
  PADDLE_ENFORCE_EQ(
      slice[1],
      3,
      common::errors::InvalidArgument(
          "slice[1]'s value is invalid, expected 3 but received %d.",
          slice[1]));

  auto drop = i.drop_front(2);
  PADDLE_ENFORCE_EQ(
      drop.size(),
      size_t(1),
      common::errors::InvalidArgument(
          "Drop's size is invalid, expected %d(size_t(1)) but received %d.",
          size_t(1),
          drop.size()));
  PADDLE_ENFORCE_EQ(
      drop[0],
      3,
      common::errors::InvalidArgument(
          "drop[0]'s value is invalid, expected 3 but received %d.", drop[0]));

  static paddle::array_ref<int> nums = {1, 2, 3, 4, 5, 6, 7, 8};
  auto front = nums.take_front(3);
  PADDLE_ENFORCE_EQ(
      front.size(),
      size_t(3),
      common::errors::InvalidArgument(
          "Front Array's size is invalid, expected %d but received %d.",
          size_t(3),
          front.size()));
  for (size_t i = 0; i < 3; ++i) {
    PADDLE_ENFORCE_EQ(
        front[i],
        nums[i],
        common::errors::InvalidArgument(
            "front[%d]'s value is invalid, expected %d but received %d.",
            i,
            nums[i],
            front[i]));
  }
  auto back = nums.take_back(3);
  PADDLE_ENFORCE_EQ(
      back.size(),
      size_t(3),
      common::errors::InvalidArgument(
          "Back Array's size is invalid, expected %d but received %d.",
          size_t(3),
          back.size()));
  for (size_t i = 0; i < 3; ++i) {
    PADDLE_ENFORCE_EQ(
        back[i],
        nums[i + 5],
        common::errors::InvalidArgument(
            "back[%d]'s value is invalid, expected %d but received %d.",
            i,
            nums[i + 5],
            back[i]));
  }
}
