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

#include <memory>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/mixed_vector.h"

template <typename T>
using vec = paddle::framework::Vector<T>;

TEST(mixed_vector, CPU_VECTOR) {
  vec<int> tmp;
  for (int i = 0; i < 10; ++i) {
    tmp.push_back(i);
  }
  ASSERT_EQ(tmp.size(), 10UL);
  vec<int> tmp2;
  tmp2 = tmp;
  ASSERT_EQ(tmp2.size(), 10UL);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(tmp2[i], i);
    ASSERT_EQ(tmp2[i], tmp[i]);
  }
  int cnt = 0;
  for (auto& t : tmp2) {
    ASSERT_EQ(t, cnt);
    ++cnt;
  }
}

TEST(mixed_vector, InitWithCount) {
  paddle::framework::Vector<int> vec(10, 10);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(vec[i], 10);
  }
}

TEST(mixed_vector, ForEach) {
  vec<int> tmp;
  for (auto& v : tmp) {
    VLOG(3) << v;
  }
}

TEST(mixed_vector, Reserve) {
  paddle::framework::Vector<int> vec;
  vec.reserve(1);
  vec.push_back(0);
  vec.push_back(0);
  vec.push_back(0);
}

TEST(mixed_vector, Resize) {
  paddle::framework::Vector<int> vec;
  vec.resize(1);
  vec.push_back(0);
  vec.push_back(0);
  vec.push_back(0);
}
