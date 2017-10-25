/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "paddle/framework/lod_tensor.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace paddle {
namespace framework {

const int kLodTensorSize = 20 * 128;

class LoDTensorTester : public ::testing::Test {
 public:
  virtual void SetUp() override {
    // tensor's batch_size: 30
    // 3 levels
    // 0 10 20
    // 0 5 10 15 20
    // 0 2 5 7 10 12 15 20
    LoD lod;
    lod.push_back(std::vector<size_t>{0, 2, 3});
    lod.push_back(std::vector<size_t>{0, 2, 5, 8});
    lod.push_back(std::vector<size_t>{0, 2, 5, 7, 10, 12, 15, 17, 20});

    ASSERT_EQ(lod.size(), 3UL);

    lod_tensor_.Resize({20 /*batch size*/, 128 /*dim*/});
    // malloc memory
    float* dst_ptr = lod_tensor_.mutable_data<float>(place);
    for (int i = 0; i < kLodTensorSize; ++i) {
      dst_ptr[i] = i;
    }

    lod_tensor_.set_lod(lod);
  }

 protected:
  platform::CPUPlace place;
  LoDTensor lod_tensor_;
};

TEST_F(LoDTensorTester, NumLevels) { ASSERT_EQ(lod_tensor_.NumLevels(), 3UL); }

TEST_F(LoDTensorTester, NumElements) {
  ASSERT_EQ(lod_tensor_.NumElements(0), 2UL);
  ASSERT_EQ(lod_tensor_.NumElements(1), 3UL);
  ASSERT_EQ(lod_tensor_.NumElements(2), 8UL);
}

TEST_F(LoDTensorTester, NumElements2) {
  ASSERT_EQ(lod_tensor_.NumElements(0, 0), 2UL);
  ASSERT_EQ(lod_tensor_.NumElements(0, 1), 1UL);
  ASSERT_EQ(lod_tensor_.NumElements(1, 1), 3UL);
}

TEST_F(LoDTensorTester, ShrinkLevels) {
  // slice 1 level
  for (size_t level = 0; level < 3UL; ++level) {
    LoDTensor new_lod_tensor = lod_tensor_;
    new_lod_tensor.ShrinkLevels(level, level + 1);
    ASSERT_EQ(new_lod_tensor.NumLevels(), 1UL);
    ASSERT_EQ(new_lod_tensor.data<float>(), lod_tensor_.data<float>());
  }
  // shrink 2 level
  for (size_t level = 0; level < 2UL; ++level) {
    LoDTensor new_lod_tensor = lod_tensor_;
    new_lod_tensor.ShrinkLevels(level, level + 2);
    // the lowest level's last element should be the tensor's batch_size.
    ASSERT_EQ(new_lod_tensor.lod().back().back(),
              lod_tensor_.lod().back().back());
    ASSERT_EQ(new_lod_tensor.NumLevels(), 2UL);
    ASSERT_EQ(new_lod_tensor.data<float>(), lod_tensor_.data<float>());
  }
}

TEST_F(LoDTensorTester, ShrinkInLevel) {
  size_t level = 0;
  LoDTensor new_lod_tensor = lod_tensor_;
  new_lod_tensor.ShrinkInLevel(level, 0, 1);
  ASSERT_EQ(new_lod_tensor.NumLevels(), 3UL);
  ASSERT_EQ(new_lod_tensor.NumElements(0), 1UL);
  ASSERT_EQ(new_lod_tensor.NumElements(1), 2UL);
  ASSERT_EQ(new_lod_tensor.NumElements(2), 5UL);
  ASSERT_EQ(new_lod_tensor.dims()[0], 12);
  for (int i = 0; i < 12 * 128; i++) {
    ASSERT_EQ(new_lod_tensor.data<float>()[i], i);
  }

  level = 1;
  new_lod_tensor = lod_tensor_;
  new_lod_tensor.ShrinkInLevel(level, 1, 2);
  ASSERT_EQ(new_lod_tensor.NumLevels(), 2UL);
  ASSERT_EQ(new_lod_tensor.NumElements(0), 1UL);
  ASSERT_EQ(new_lod_tensor.NumElements(1), 3UL);
  ASSERT_EQ(new_lod_tensor.dims()[0], 7);
  for (int i = 5 * 128; i < 12 * 128; i++) {
    ASSERT_EQ(new_lod_tensor.data<float>()[i - 5 * 128], i);
  }

  LoDTensor t1;
  t1.set_lod(lod_tensor_.lod());
  t1.ShareDataWith(lod_tensor_);

  LoDTensor t2;
  t2.set_lod(lod_tensor_.lod());
  t2.ShareDataWith(lod_tensor_);

  t1.ShrinkInLevel(0, 1, 2);
  t2.ShrinkInLevel(0, 0, 1);
  EXPECT_NE(t1.data<float>(), t2.data<float>());
  EXPECT_NE(t1.data<float>(), lod_tensor_.data<float>());
}

TEST(LodExpand, test) {
  LoD lod{{0, 2}};
  LoDTensor tensor;
  tensor.set_lod(lod);
  tensor.Resize({2, 1});
  tensor.mutable_data<float>(platform::CPUPlace());
  tensor.data<float>()[0] = 0;
  tensor.data<float>()[1] = 1;

  LoD target;
  target.emplace_back(std::vector<size_t>{0, 3, 5});
  auto new_tensor = LodExpand<float>(tensor, target, 0UL, platform::CPUPlace());
  std::vector<int> result{{0, 0, 0, 1, 1}};
  for (size_t i = 0; i < 5; i++) {
    ASSERT_EQ(new_tensor.data<float>()[i], result[i]);
  }
}

TEST_F(LoDTensorTester, SerializeDeserialize) {
  LoDTensor new_lod_tensor = lod_tensor_;
  float* src_ptr = lod_tensor_.data<float>();
  std::string s = lod_tensor_.SerializeToString();
  LoDTensor dst;
  dst.DeserializeFromString(s, platform::CPUPlace());
  float* dst_ptr = dst.data<float>();
  for (int i = 0; i < kLodTensorSize; ++i) {
    EXPECT_EQ(dst_ptr[i], src_ptr[i]);
  }

  ASSERT_EQ(dst.NumElements(0), 2UL);
  ASSERT_EQ(dst.NumElements(1), 3UL);
  ASSERT_EQ(dst.NumElements(2), 8UL);
}

}  // namespace framework
}  // namespace paddle
