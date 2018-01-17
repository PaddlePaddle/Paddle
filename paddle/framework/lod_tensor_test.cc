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

TEST(LoD, GetFineGrainedLoDLength) {
  LoD lod;
  lod.push_back(std::vector<size_t>({0, 2, 4, 5}));
  lod.push_back(std::vector<size_t>({0, 1, 6, 8, 10, 11}));
  lod.push_back(
      std::vector<size_t>({0, 2, 5, 7, 10, 12, 15, 17, 20, 24, 26, 29}));

  auto lod_and_offset =
      paddle::framework::GetSubLoDAndAbsoluteOffset(lod, 1, 2, 0);
  LoD lod_length = lod_and_offset.first;
  size_t start_offset = lod_and_offset.second.first;
  size_t end_offset = lod_and_offset.second.second;

  LoD expected;
  expected.push_back(std::vector<size_t>{2});
  expected.push_back(std::vector<size_t>{2, 2});
  expected.push_back(std::vector<size_t>{2, 3, 4, 2});
  EXPECT_EQ(lod_length, expected);
  EXPECT_EQ(start_offset, 15UL);
  EXPECT_EQ(end_offset, 26UL);
}

TEST(LoD, AppendLoD) {
  LoD lod_lens;
  lod_lens.push_back(std::vector<size_t>({2}));
  lod_lens.push_back(std::vector<size_t>({2, 2}));
  lod_lens.push_back(std::vector<size_t>({2, 3, 4, 2}));

  LoD origin;
  origin.push_back(std::vector<size_t>({0, 2}));
  origin.push_back(std::vector<size_t>({0, 1, 6}));
  origin.push_back(std::vector<size_t>({0, 2, 5, 7, 10, 12, 15}));

  paddle::framework::AppendLoD(&origin, lod_lens);

  LoD expected;
  expected.push_back(std::vector<size_t>({0, 2, 4}));
  expected.push_back(std::vector<size_t>({0, 1, 6, 8, 10}));
  expected.push_back(
      std::vector<size_t>({0, 2, 5, 7, 10, 12, 15, 17, 20, 24, 26}));
  EXPECT_EQ(origin, expected);
}

TEST(LoD, ToAbsOffset) {
  LoD relative_lod;
  relative_lod.push_back(std::vector<size_t>({0, 2}));
  relative_lod.push_back(std::vector<size_t>({0, 1, 3}));
  relative_lod.push_back(std::vector<size_t>({0, 2, 4, 5}));

  LoD abs_lod = paddle::framework::ToAbsOffset(relative_lod);

  LoD expected;
  expected.push_back(std::vector<size_t>({0, 5}));
  expected.push_back(std::vector<size_t>({0, 2, 5}));
  expected.push_back(std::vector<size_t>({0, 2, 4, 5}));

  EXPECT_EQ(abs_lod, expected);
}

TEST(LoD, SplitLoDTensor) {
  LoD lod;
  lod.push_back(std::vector<size_t>({0, 2, 4, 5, 6}));
  lod.push_back(std::vector<size_t>({0, 1, 6, 8, 13, 15, 20}));

  platform::CPUPlace place;
  LoDTensor lod_tensor;
  lod_tensor.Resize({20, 1});
  float* dst_ptr = lod_tensor.mutable_data<float>(place);
  for (int i = 0; i < lod_tensor.numel(); ++i) {
    dst_ptr[i] = i;
  }
  lod_tensor.set_lod(lod);

  std::vector<platform::Place> places{platform::CPUPlace(),
                                      platform::CPUPlace()};
  LoD lod0;
  lod0.push_back(std::vector<size_t>({0, 2, 4}));
  lod0.push_back(std::vector<size_t>({0, 1, 6, 8, 13}));
  LoD lod1;
  lod1.push_back(std::vector<size_t>({0, 1, 2}));
  lod1.push_back(std::vector<size_t>({0, 2, 7}));

  auto lods = lod_tensor.SplitLoDTensor(places);
  EXPECT_EQ(lods[0].lod(), lod0);
  EXPECT_EQ(lods[1].lod(), lod1);
}

TEST(LoD, MergeLoDTensor) {
  LoD lod;
  lod.push_back(std::vector<size_t>({0, 2, 4, 5, 6}));
  lod.push_back(std::vector<size_t>({0, 1, 6, 8, 13, 15, 20}));

  platform::CPUPlace place;

  LoDTensor lod_tensor0;
  LoD lod0;
  lod0.push_back(std::vector<size_t>({0, 2, 4}));
  lod0.push_back(std::vector<size_t>({0, 1, 6, 8, 13}));
  lod_tensor0.set_lod(lod0);

  lod_tensor0.Resize({13, 1});
  float* dst_ptr = lod_tensor0.mutable_data<float>(place);
  for (int i = 0; i < lod_tensor0.numel(); ++i) {
    dst_ptr[i] = i;
  }

  LoDTensor lod_tensor1;
  LoD lod1;
  lod1.push_back(std::vector<size_t>({0, 1, 2}));
  lod1.push_back(std::vector<size_t>({0, 2, 7}));
  lod_tensor1.set_lod(lod1);
  lod_tensor1.Resize({7, 1});
  dst_ptr = lod_tensor1.mutable_data<float>(place);
  for (int i = 0; i < lod_tensor1.numel(); ++i) {
    dst_ptr[i] = i;
  }

  std::vector<const LoDTensor*> lods{&lod_tensor0, &lod_tensor1};

  LoDTensor lod_tensor;
  lod_tensor.MergeLoDTensor(lods, place);
  EXPECT_EQ(lod_tensor.lod(), lod);
}

}  // namespace framework
}  // namespace paddle
