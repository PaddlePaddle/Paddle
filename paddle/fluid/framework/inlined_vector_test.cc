// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/inlined_vector.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"

namespace paddle {
namespace framework {

template <typename T, size_t N>
static std::vector<T> ToStdVector(const framework::InlinedVector<T, N> &vec) {
  std::vector<T> std_vec;
  std_vec.reserve(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    std_vec.emplace_back(vec[i]);
  }
  return std_vec;
}

template <size_t N>
void InlinedVectorCheck(size_t n) {
  std::srand(std::time(nullptr));

  std::vector<int> std_vec;
  framework::InlinedVector<int, N> vec;

  for (size_t i = 0; i < n; ++i) {
    int value = rand();  // NOLINT

    std_vec.emplace_back(value);
    vec.emplace_back(value);

    CHECK_EQ(std_vec.size(), vec.size());
    CHECK_EQ(std_vec.back(), vec.back());

    CHECK_EQ(vec.back(), value);
  }

  bool is_equal = (std_vec == ToStdVector(vec));

  CHECK_EQ(is_equal, true);

  for (size_t i = 0; i < n; ++i) {
    CHECK_EQ(std_vec.size(), vec.size());
    CHECK_EQ(std_vec.back(), vec.back());
    std_vec.pop_back();
    vec.pop_back();
    CHECK_EQ(std_vec.size(), vec.size());
  }

  CHECK_EQ(std_vec.size(), static_cast<size_t>(0));
  CHECK_EQ(vec.size(), static_cast<size_t>(0));
}

TEST(inlined_vector, inlined_vector) {
  for (size_t i = 0; i < 20; ++i) {
    InlinedVectorCheck<1>(i);
    InlinedVectorCheck<10>(i);
    InlinedVectorCheck<15>(i);
    InlinedVectorCheck<20>(i);
    InlinedVectorCheck<21>(i);
    InlinedVectorCheck<25>(i);
  }
}

}  // namespace framework
}  // namespace paddle
