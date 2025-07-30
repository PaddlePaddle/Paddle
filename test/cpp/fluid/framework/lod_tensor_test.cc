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

#include "paddle/fluid/framework/lod_tensor.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/phi/core/lod_utils.h"

namespace paddle {
namespace framework {

TEST(LegacyLoD, PrintDenseTensor) {
  phi::DenseTensor tensor1;
  tensor1.Resize({2});
  tensor1.mutable_data<float>(phi::CPUPlace());
  tensor1.data<float>()[0] = 0.2;
  tensor1.data<float>()[1] = 0.5;
  LOG(INFO) << tensor1;

  phi::DenseTensor tensor2;
  tensor2.Resize({2});
  tensor2.mutable_data<int64_t>(phi::CPUPlace());
  tensor2.data<int64_t>()[0] = 1;
  tensor2.data<int64_t>()[1] = 2;
  LOG(INFO) << tensor2;
}

TEST(LegacyLoD, data) {
  LegacyLoD lod{{0, 1, 2}};
  lod.push_back({0, 2, 4, 5});
  lod.push_back(std::vector<size_t>({0, 1, 6, 8, 10, 11}));

  auto& v = lod[0];
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v[i], i);
  }
}

TEST(LegacyLoD, AppendLegacyLoD) {
  LegacyLoD lod_lens;
  lod_lens.push_back(std::vector<size_t>({2}));
  lod_lens.push_back(std::vector<size_t>({2, 2}));
  lod_lens.push_back(std::vector<size_t>({2, 3, 4, 2}));

  LegacyLoD origin;
  origin.push_back(std::vector<size_t>({0, 2}));
  origin.push_back(std::vector<size_t>({0, 1, 6}));
  origin.push_back(std::vector<size_t>({0, 2, 5, 7, 10, 12, 15}));

  phi::AppendLegacyLoD(&origin, lod_lens);

  LegacyLoD expected;
  expected.push_back(std::vector<size_t>({0, 2, 4}));
  expected.push_back(std::vector<size_t>({0, 1, 6, 8, 10}));
  expected.push_back(
      std::vector<size_t>({0, 2, 5, 7, 10, 12, 15, 17, 20, 24, 26}));
  EXPECT_EQ(origin, expected);
}

TEST(LegacyLoD, CheckLegacyLoD) {
  LegacyLoD relative_lod;
  relative_lod.push_back(std::vector<size_t>({0, 2}));
  relative_lod.push_back(std::vector<size_t>({0, 1, 3}));
  relative_lod.push_back(std::vector<size_t>({0, 2, 4, 5}));

  // check compatible
  ASSERT_TRUE(CheckLegacyLoD(relative_lod));
  relative_lod[1].back()++;
  ASSERT_FALSE(CheckLegacyLoD(relative_lod));
  relative_lod[1].back()--;  // recover it

  // check empty
  LegacyLoD empty_lod;
  ASSERT_TRUE(CheckLegacyLoD(empty_lod));

  // check less than 2 offsets in a level
  LegacyLoD some_lod0;
  some_lod0.push_back(std::vector<size_t>({0}));
  ASSERT_FALSE(CheckLegacyLoD(some_lod0));

  // check with underlying tensor storage.
  ASSERT_TRUE(CheckLegacyLoD(relative_lod, 5));
  ASSERT_FALSE(CheckLegacyLoD(relative_lod, 9));

  // check whether lod is ascending-sorted (allow same items)
  ASSERT_TRUE(CheckLegacyLoD({{0, 1, 2, 3, 4, 5}}, 5));
  ASSERT_TRUE(CheckLegacyLoD({{0, 1, 3, 3, 4, 5}}, 5));
  ASSERT_FALSE(CheckLegacyLoD({{0, 1, 3, 2, 5}}, 5));
}

TEST(LegacyLoD, ConvertToLengthBasedLegacyLoD) {
  LegacyLoD offset_lod;
  offset_lod.push_back(std::vector<size_t>({0, 2}));
  offset_lod.push_back(std::vector<size_t>({0, 1, 3}));
  offset_lod.push_back(std::vector<size_t>({0, 2, 4, 5}));

  LegacyLoD length_lod = phi::ConvertToLengthBasedLegacyLoD(offset_lod);

  LegacyLoD expected;
  expected.push_back(std::vector<size_t>({2}));
  expected.push_back(std::vector<size_t>({1, 2}));
  expected.push_back(std::vector<size_t>({2, 2, 1}));

  EXPECT_EQ(length_lod, expected);
}

TEST(LegacyLoD, ConvertToOffsetBasedLegacyLoD) {
  LegacyLoD length_lod;
  length_lod.push_back(std::vector<size_t>({2}));
  length_lod.push_back(std::vector<size_t>({1, 2}));
  length_lod.push_back(std::vector<size_t>({2, 2, 1}));

  LegacyLoD offset_lod = ConvertToOffsetBasedLegacyLoD(length_lod);

  LegacyLoD expected;
  expected.push_back(std::vector<size_t>({0, 2}));
  expected.push_back(std::vector<size_t>({0, 1, 3}));
  expected.push_back(std::vector<size_t>({0, 2, 4, 5}));

  EXPECT_EQ(offset_lod, expected);
}

}  // namespace framework
}  // namespace paddle
