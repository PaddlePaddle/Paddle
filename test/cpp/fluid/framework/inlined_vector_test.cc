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

#include "glog/logging.h"
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

    PADDLE_ENFORCE_EQ(std_vec.size(),
                      vec.size(),
                      phi::errors::InvalidArgument(
                          "The sizes of std_vec and vec should be equal, but "
                          "received std_vec.size() = %d and vec.size() = %d.",
                          std_vec.size(),
                          vec.size()));
    PADDLE_ENFORCE_EQ(
        std_vec.back(),
        vec.back(),
        phi::errors::InvalidArgument(
            "The last elements of std_vec and vec should be equal, but "
            "received std_vec.back() = %d and vec.back() = %d.",
            std_vec.back(),
            vec.back()));

    PADDLE_ENFORCE_EQ(vec.back(),
                      value,
                      phi::errors::InvalidArgument(
                          "The last element of vec should be equal to value, "
                          "but received vec.back() = %d and value = %d.",
                          vec.back(),
                          value));
  }

  bool is_equal = (std_vec == ToStdVector(vec));

  PADDLE_ENFORCE_EQ(
      is_equal,
      true,
      phi::errors::InvalidArgument(
          "The std_vec and vec should be equal, but they are not."));

  for (size_t i = 0; i < n; ++i) {
    PADDLE_ENFORCE_EQ(std_vec.size(),
                      vec.size(),
                      phi::errors::InvalidArgument(
                          "The sizes of std_vec and vec should be equal, but "
                          "received std_vec.size() = %d and vec.size() = %d.",
                          std_vec.size(),
                          vec.size()));
    PADDLE_ENFORCE_EQ(
        std_vec.back(),
        vec.back(),
        phi::errors::InvalidArgument(
            "The last elements of std_vec and vec should be equal, but "
            "received std_vec.back() = %d and vec.back() = %d.",
            std_vec.back(),
            vec.back()));
    std_vec.pop_back();
    vec.pop_back();
    PADDLE_ENFORCE_EQ(std_vec.size(),
                      vec.size(),
                      phi::errors::InvalidArgument(
                          "The sizes of std_vec and vec should be equal, but "
                          "received std_vec.size() = %d and vec.size() = %d.",
                          std_vec.size(),
                          vec.size()));
  }

  PADDLE_ENFORCE_EQ(std_vec.size(),
                    vec.size(),
                    phi::errors::InvalidArgument(
                        "The sizes of std_vec and vec should be equal, but "
                        "received std_vec.size() = %d and vec.size() = %d.",
                        std_vec.size(),
                        vec.size()));
  PADDLE_ENFORCE_EQ(
      vec.size(),
      static_cast<size_t>(0),
      phi::errors::InvalidArgument(
          "The size of vec should be 0, but received vec.size() = %d.",
          vec.size()));
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
