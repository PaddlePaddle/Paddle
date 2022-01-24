//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {

TEST(LoD, PrintLoDTensor) {
  LoDTensor tensor1;
  tensor1.Resize({2});
  tensor1.mutable_data<float>(platform::CPUPlace());
  tensor1.data<float>()[0] = 0.2;
  tensor1.data<float>()[1] = 0.5;
  LOG(INFO) << tensor1;

  LoDTensor tensor2;
  tensor2.Resize({2});
  tensor2.mutable_data<int64_t>(platform::CPUPlace());
  tensor2.data<int64_t>()[0] = 1;
  tensor2.data<int64_t>()[1] = 2;
  LOG(INFO) << tensor2;
}

TEST(LoD, data) {
  LoD lod{{0, 1, 2}};
  lod.push_back({0, 2, 4, 5});
  lod.push_back(std::vector<size_t>({0, 1, 6, 8, 10, 11}));

  auto& v = lod[0];
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v[i], i);
  }
}

TEST(LoD, ExpandLoD) {
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

  auto lods = SplitLoDTensor(lod_tensor, places);
  EXPECT_EQ(lods[0].lod(), lod0);
  EXPECT_EQ(lods[1].lod(), lod1);
}

TEST(LoD, SplitLoDTensorWithZeroBatchSize) {
  LoD lod;
  lod.push_back(std::vector<size_t>({0}));

  platform::CPUPlace place;
  LoDTensor lod_tensor;
  lod_tensor.Resize({0, 5});
  lod_tensor.mutable_data<float>(place);
  lod_tensor.set_lod(lod);

  std::vector<platform::Place> places{platform::CPUPlace(),
                                      platform::CPUPlace()};
  LoD lod_res;
  lod_res.push_back(std::vector<size_t>({0}));

  auto lods = SplitLoDTensor(lod_tensor, places);
  EXPECT_EQ(lods[0].lod(), lod_res);
  EXPECT_EQ(lods[1].lod(), lod_res);
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

  LoDTensor lod_tensor2;
  LoD lod2;
  lod2.push_back(std::vector<size_t>({0}));
  lod2.push_back(std::vector<size_t>({0}));
  lod_tensor2.set_lod(lod2);
  lod_tensor2.Resize({0});
  dst_ptr = lod_tensor2.mutable_data<float>(place);

  std::vector<const LoDTensor*> lods{&lod_tensor0, &lod_tensor1, &lod_tensor2};

  LoDTensor lod_tensor;
  MergeLoDTensor(&lod_tensor, lods, place);
  EXPECT_EQ(lod_tensor.lod(), lod);
}

TEST(LoD, CheckLoD) {
  LoD relative_lod;
  relative_lod.push_back(std::vector<size_t>({0, 2}));
  relative_lod.push_back(std::vector<size_t>({0, 1, 3}));
  relative_lod.push_back(std::vector<size_t>({0, 2, 4, 5}));

  // check compatible
  ASSERT_TRUE(CheckLoD(relative_lod));
  relative_lod[1].back()++;
  ASSERT_FALSE(CheckLoD(relative_lod));
  relative_lod[1].back()--;  // recover it

  // check empty
  LoD empty_lod;
  ASSERT_TRUE(CheckLoD(empty_lod));

  // check less than 2 offsets in a level
  LoD some_lod0;
  some_lod0.push_back(std::vector<size_t>({0}));
  ASSERT_FALSE(CheckLoD(some_lod0));

  // check with underlying tensor storage.
  ASSERT_TRUE(CheckLoD(relative_lod, 5));
  ASSERT_FALSE(CheckLoD(relative_lod, 9));

  // check whether lod is ascending-sorted (allow same items)
  ASSERT_TRUE(CheckLoD({{0, 1, 2, 3, 4, 5}}, 5));
  ASSERT_TRUE(CheckLoD({{0, 1, 3, 3, 4, 5}}, 5));
  ASSERT_FALSE(CheckLoD({{0, 1, 3, 2, 5}}, 5));
}

TEST(LoD, CheckAbsLoD) {
  LoD relative_lod;
  relative_lod.push_back(std::vector<size_t>({0, 2}));
  relative_lod.push_back(std::vector<size_t>({0, 1, 3}));
  relative_lod.push_back(std::vector<size_t>({0, 2, 4, 5}));

  auto abs_lod = ToAbsOffset(relative_lod);

  ASSERT_TRUE(CheckAbsLoD(abs_lod));

  // check less than 2 offsets in a level.

  // check the last item should be compatible with tensor height.
  abs_lod.back().back()++;
  ASSERT_FALSE(CheckAbsLoD(abs_lod));
  abs_lod.back().back()--;  // restore

  // check less than 2 offsets in a lod.
  LoD abs_lod0;
  abs_lod0.push_back(std::vector<size_t>({0}));
  ASSERT_FALSE(CheckAbsLoD(abs_lod0));
}

TEST(LoD, ConvertToLengthBasedLoD) {
  LoD offset_lod;
  offset_lod.push_back(std::vector<size_t>({0, 2}));
  offset_lod.push_back(std::vector<size_t>({0, 1, 3}));
  offset_lod.push_back(std::vector<size_t>({0, 2, 4, 5}));

  LoD length_lod = ConvertToLengthBasedLoD(offset_lod);

  LoD expected;
  expected.push_back(std::vector<size_t>({2}));
  expected.push_back(std::vector<size_t>({1, 2}));
  expected.push_back(std::vector<size_t>({2, 2, 1}));

  EXPECT_EQ(length_lod, expected);
}

TEST(LoD, ConvertToOffsetBasedLoD) {
  LoD length_lod;
  length_lod.push_back(std::vector<size_t>({2}));
  length_lod.push_back(std::vector<size_t>({1, 2}));
  length_lod.push_back(std::vector<size_t>({2, 2, 1}));

  LoD offset_lod = ConvertToOffsetBasedLoD(length_lod);

  LoD expected;
  expected.push_back(std::vector<size_t>({0, 2}));
  expected.push_back(std::vector<size_t>({0, 1, 3}));
  expected.push_back(std::vector<size_t>({0, 2, 4, 5}));

  EXPECT_EQ(offset_lod, expected);
}

}  // namespace framework
}  // namespace paddle
