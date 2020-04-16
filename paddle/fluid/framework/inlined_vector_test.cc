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

TEST(inlined_vector, test_emplace_back_and_pop_back) {
  for (size_t i = 0; i < 20; ++i) {
    InlinedVectorCheck<1>(i);
    InlinedVectorCheck<10>(i);
    InlinedVectorCheck<15>(i);
    InlinedVectorCheck<20>(i);
    InlinedVectorCheck<21>(i);
    InlinedVectorCheck<25>(i);
  }
}

class StubObject {
 public:
  StubObject() { ++alived_obj_num_; }

  StubObject(const StubObject &other) {
    rand_num_ = other.rand_num_;
    ++alived_obj_num_;
  }

  StubObject(StubObject &&other) {
    rand_num_ = other.rand_num_;
    other.rand_num_ = -1;
    ++alived_obj_num_;
  }

  ~StubObject() { --alived_obj_num_; }

  StubObject &operator=(const StubObject &other) {
    rand_num_ = other.rand_num_;
    return *this;
  }

  StubObject &operator=(StubObject &&other) {
    rand_num_ = other.rand_num_;
    other.rand_num_ = -1;
    return *this;
  }

  static size_t AlivedObjectNum() { return alived_obj_num_; }

  void SetRandomNumber() { rand_num_ = rand(); }  // NOLINT

  int RandomNumber() const { return rand_num_; }

  bool operator==(const StubObject &other) const {
    return rand_num_ == other.rand_num_;
  }

 private:
  int rand_num_;
  static size_t alived_obj_num_;
};

size_t StubObject::alived_obj_num_ = 0;

static std::ostream &operator<<(std::ostream &out, const StubObject &obj) {
  out << obj.RandomNumber();
  return out;
}

template <size_t N>
static void TestInlinedVectorCopyAndMove() {
  using VecType = InlinedVector<StubObject, N>;
  VecType vec;
  ASSERT_EQ(vec.size(), 0UL);
  ASSERT_EQ(vec.empty(), true);

  for (size_t i = 0; i < 3 * N; ++i) {
    vec.resize(i);
    for (size_t j = 0; j < i; ++j) {
      vec[j].SetRandomNumber();
    }

    ASSERT_EQ(vec.size(), i);
    ASSERT_EQ(vec.empty(), (i == 0));
    {
      auto copy_vec = vec;
      ASSERT_EQ(copy_vec.size(), vec.size());
      ASSERT_EQ(StubObject::AlivedObjectNum(), 2 * i);
      for (size_t j = 0; j < i; ++j) {
        ASSERT_EQ(vec[j], copy_vec[j]);
      }

      {
        VecType another_copy_vec(rand() % 100);  // NOLINT
        another_copy_vec = vec;
        ASSERT_EQ(another_copy_vec.size(), vec.size());
        ASSERT_EQ(StubObject::AlivedObjectNum(), 3 * i);
        for (size_t j = 0; j < i; ++j) {
          ASSERT_EQ(vec[j], another_copy_vec[j]);
        }
      }

      auto move_vec = std::move(copy_vec);
      ASSERT_EQ(copy_vec.size(), 0UL);
      ASSERT_EQ(move_vec.size(), vec.size());
      ASSERT_EQ(StubObject::AlivedObjectNum(), 2 * i);
      for (size_t j = 0; j < i; ++j) {
        ASSERT_EQ(vec[j], move_vec[j]);
      }

      {
        VecType another_move_vec(rand() % 100);  // NOLINT
        another_move_vec = std::move(move_vec);
        ASSERT_EQ(another_move_vec.size(), vec.size());
        ASSERT_EQ(move_vec.empty(), true);
        ASSERT_EQ(StubObject::AlivedObjectNum(), 2 * i);
        for (size_t j = 0; j < i; ++j) {
          ASSERT_EQ(vec[j], another_move_vec[j]);
        }
      }
    }

    ASSERT_EQ(StubObject::AlivedObjectNum(), i);
  }
}

TEST(inlined_vector, test_copy_and_move) {
  TestInlinedVectorCopyAndMove<1>();
  TestInlinedVectorCopyAndMove<10>();
  TestInlinedVectorCopyAndMove<15>();
  TestInlinedVectorCopyAndMove<20>();
  TestInlinedVectorCopyAndMove<21>();
  TestInlinedVectorCopyAndMove<25>();
}

TEST(inlined_vector, test_clear) {
  InlinedVector<StubObject, 8> vec;
  for (size_t i = 0; i < 20; ++i) {
    vec.resize(i);
    ASSERT_EQ(StubObject::AlivedObjectNum(), i);
    vec.clear();
    ASSERT_EQ(StubObject::AlivedObjectNum(), 0UL);
  }
}

TEST(inlined_vector, test_front_back) {
  InlinedVector<int, 3> vec = {4, 3, 2, 10};
  ASSERT_EQ(vec.front(), 4);
  ASSERT_EQ(vec.back(), 10);

  ASSERT_EQ(vec[0], vec.front());
  ASSERT_EQ(vec[vec.size() - 1], vec.back());

  const auto const_vec = vec;
  ASSERT_EQ(const_vec.front(), vec[0]);
  ASSERT_EQ(const_vec.back(), vec[vec.size() - 1]);
}

}  // namespace framework
}  // namespace paddle
