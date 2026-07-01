// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "test/cpp/prim/init_env_utils.h"
#include "torch/all.h"

namespace {

class ScatterReduceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { paddle::prim::InitTensorOperants(); }
};

}  // namespace

TEST_F(ScatterReduceTest, ScatterReduceSum) {
  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "sum");

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 5);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);   // result[0,0]
  ASSERT_FLOAT_EQ(data[1], 0.0f);   // result[0,1]
  ASSERT_FLOAT_EQ(data[2], 0.0f);   // result[0,2]
  ASSERT_FLOAT_EQ(data[3], 4.0f);   // result[0,3]
  ASSERT_FLOAT_EQ(data[4], 5.0f);   // result[0,4]
  ASSERT_FLOAT_EQ(data[5], 0.0f);   // result[1,0]
  ASSERT_FLOAT_EQ(data[6], 2.0f);   // result[1,1]
  ASSERT_FLOAT_EQ(data[7], 0.0f);   // result[1,2]
  ASSERT_FLOAT_EQ(data[8], 0.0f);   // result[1,3]
  ASSERT_FLOAT_EQ(data[9], 0.0f);   // result[1,4]
  ASSERT_FLOAT_EQ(data[10], 0.0f);  // result[2,0]
  ASSERT_FLOAT_EQ(data[11], 0.0f);  // result[2,1]
  ASSERT_FLOAT_EQ(data[12], 3.0f);  // result[2,2]
  ASSERT_FLOAT_EQ(data[13], 0.0f);  // result[2,3]
  ASSERT_FLOAT_EQ(data[14], 0.0f);  // result[2,4]
}

TEST_F(ScatterReduceTest, ScatterReduceFreeFunctionSum) {
  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;

  at::Tensor result = at::scatter_reduce(self, 0, index, src, "sum");

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[3], 4.0f);
  ASSERT_FLOAT_EQ(data[4], 5.0f);
  ASSERT_FLOAT_EQ(data[6], 2.0f);
  ASSERT_FLOAT_EQ(data[12], 3.0f);
}

TEST_F(ScatterReduceTest, ScatterReduceReplace) {
  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;

  EXPECT_THROW(self.scatter_reduce(0, index, src, "replace"),
               std::invalid_argument);
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceSum) {
  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;

  self.scatter_reduce_(0, index, src, "sum");

  float* data = self.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);   // self[0,0]
  ASSERT_FLOAT_EQ(data[6], 2.0f);   // self[1,1]
  ASSERT_FLOAT_EQ(data[12], 3.0f);  // self[2,2]
  ASSERT_FLOAT_EQ(data[3], 4.0f);   // self[0,3]
  ASSERT_FLOAT_EQ(data[4], 5.0f);   // self[0,4]
}

TEST_F(ScatterReduceTest, ScatterReduceDim1) {
  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::full({2, 4}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  src.data_ptr<float>()[5] = 6.0f;
  src.data_ptr<float>()[6] = 7.0f;
  src.data_ptr<float>()[7] = 8.0f;

  at::Tensor result = self.scatter_reduce(1, index, src, "sum");

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 4);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);  // result[0,0]
  ASSERT_FLOAT_EQ(data[1], 6.0f);  // result[0,1] = 2+4
  ASSERT_FLOAT_EQ(data[2], 3.0f);  // result[0,2]
  ASSERT_FLOAT_EQ(data[3], 0.0f);  // result[0,3]
  ASSERT_FLOAT_EQ(data[4], 6.0f);  // result[1,0]
  ASSERT_FLOAT_EQ(data[5], 7.0f);  // result[1,1]
  ASSERT_FLOAT_EQ(data[6], 8.0f);  // result[1,2]
  ASSERT_FLOAT_EQ(data[7], 5.0f);  // result[1,3]
}

TEST_F(ScatterReduceTest, ScatterReduceNegativeDim) {
  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::full({2, 4}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  src.data_ptr<float>()[5] = 6.0f;
  src.data_ptr<float>()[6] = 7.0f;
  src.data_ptr<float>()[7] = 8.0f;

  at::Tensor result = self.scatter_reduce(-1, index, src, "sum");

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 4);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[1], 6.0f);
  ASSERT_FLOAT_EQ(data[2], 3.0f);
  ASSERT_FLOAT_EQ(data[3], 0.0f);
  ASSERT_FLOAT_EQ(data[4], 6.0f);
  ASSERT_FLOAT_EQ(data[5], 7.0f);
  ASSERT_FLOAT_EQ(data[6], 8.0f);
  ASSERT_FLOAT_EQ(data[7], 5.0f);
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceNegativeDim) {
  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::full({2, 4}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  src.data_ptr<float>()[5] = 6.0f;
  src.data_ptr<float>()[6] = 7.0f;
  src.data_ptr<float>()[7] = 8.0f;

  self.scatter_reduce_(-1, index, src, "sum");

  float* data = self.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[1], 6.0f);
  ASSERT_FLOAT_EQ(data[2], 3.0f);
  ASSERT_FLOAT_EQ(data[3], 0.0f);
  ASSERT_FLOAT_EQ(data[4], 6.0f);
  ASSERT_FLOAT_EQ(data[5], 7.0f);
  ASSERT_FLOAT_EQ(data[6], 8.0f);
  ASSERT_FLOAT_EQ(data[7], 5.0f);
}

TEST_F(ScatterReduceTest, ScatterReduceAmax) {
  // Use {2, 3} src/index with duplicate target positions to verify
  // true max-reduce aggregation behavior.
  at::Tensor self = at::zeros({2, 3}, at::kFloat);
  at::Tensor index = at::zeros({2, 3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 0;
  index.data_ptr<int64_t>()[2] = 1;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 1;
  index.data_ptr<int64_t>()[5] = 1;
  at::Tensor src = at::full({2, 3}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 100.0f;
  src.data_ptr<float>()[4] = 200.0f;
  src.data_ptr<float>()[5] = 300.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "amax");

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 3);
  float* data = result.data_ptr<float>();
  // result[0,0]: src[0,0]=10, src[1,0]=100 -> amax = 100
  ASSERT_FLOAT_EQ(data[0], 100.0f);
  // result[0,1]: src[0,1]=20 -> amax = 20
  ASSERT_FLOAT_EQ(data[1], 20.0f);
  // result[0,2]: no src maps here -> 0
  ASSERT_FLOAT_EQ(data[2], 0.0f);
  // result[1,0]: no src maps here -> 0
  ASSERT_FLOAT_EQ(data[3], 0.0f);
  // result[1,1]: src[1,1]=200 -> amax = 200
  ASSERT_FLOAT_EQ(data[4], 200.0f);
  // result[1,2]: src[0,2]=30, src[1,2]=300 -> amax = 300
  ASSERT_FLOAT_EQ(data[5], 300.0f);
}

TEST_F(ScatterReduceTest, ScatterReduceProd) {
  at::Tensor self = at::ones({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 2.0f;
  src.data_ptr<float>()[1] = 3.0f;
  src.data_ptr<float>()[2] = 4.0f;
  src.data_ptr<float>()[3] = 5.0f;
  src.data_ptr<float>()[4] = 6.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "prod");

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 5);
  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);   // result[0,0] = 1*2
  ASSERT_FLOAT_EQ(data[1], 1.0f);   // result[0,1] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[2], 1.0f);   // result[0,2] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[3], 5.0f);   // result[0,3] = 1*5
  ASSERT_FLOAT_EQ(data[4], 6.0f);   // result[0,4] = 1*6
  ASSERT_FLOAT_EQ(data[5], 1.0f);   // result[1,0] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[6], 3.0f);   // result[1,1] = 1*3
  ASSERT_FLOAT_EQ(data[7], 1.0f);   // result[1,2] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[8], 1.0f);   // result[1,3] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[9], 1.0f);   // result[1,4] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[10], 1.0f);  // result[2,0] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[11], 1.0f);  // result[2,1] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[12], 4.0f);  // result[2,2] = 1*4
  ASSERT_FLOAT_EQ(data[13], 1.0f);  // result[2,3] (unchanged, self=1)
  ASSERT_FLOAT_EQ(data[14], 1.0f);  // result[2,4] (unchanged, self=1)
}

TEST_F(ScatterReduceTest, ScatterReduceMean) {
  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "mean");

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 5);
  float* data = result.data_ptr<float>();
  // mean with include_self: (self + src) / 2 for mapped positions
  ASSERT_FLOAT_EQ(data[0], 5.0f);    // result[0,0] = (0+10)/2
  ASSERT_FLOAT_EQ(data[1], 0.0f);    // result[0,1]
  ASSERT_FLOAT_EQ(data[2], 0.0f);    // result[0,2]
  ASSERT_FLOAT_EQ(data[3], 20.0f);   // result[0,3] = (0+40)/2
  ASSERT_FLOAT_EQ(data[4], 25.0f);   // result[0,4] = (0+50)/2
  ASSERT_FLOAT_EQ(data[5], 0.0f);    // result[1,0]
  ASSERT_FLOAT_EQ(data[6], 10.0f);   // result[1,1] = (0+20)/2
  ASSERT_FLOAT_EQ(data[7], 0.0f);    // result[1,2]
  ASSERT_FLOAT_EQ(data[8], 0.0f);    // result[1,3]
  ASSERT_FLOAT_EQ(data[9], 0.0f);    // result[1,4]
  ASSERT_FLOAT_EQ(data[10], 0.0f);   // result[2,0]
  ASSERT_FLOAT_EQ(data[11], 0.0f);   // result[2,1]
  ASSERT_FLOAT_EQ(data[12], 15.0f);  // result[2,2] = (0+30)/2
  ASSERT_FLOAT_EQ(data[13], 0.0f);   // result[2,3]
  ASSERT_FLOAT_EQ(data[14], 0.0f);   // result[2,4]
}

TEST_F(ScatterReduceTest, ScatterReduceAmin) {
  at::Tensor self = at::full({3, 5}, 100.0f, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "amin");

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 5);
  float* data = result.data_ptr<float>();
  // amin with include_self: min(self, src) for mapped positions
  ASSERT_FLOAT_EQ(data[0], 10.0f);    // result[0,0] = min(100,10)
  ASSERT_FLOAT_EQ(data[1], 100.0f);   // result[0,1] = 100 (no src)
  ASSERT_FLOAT_EQ(data[2], 100.0f);   // result[0,2] = 100 (no src)
  ASSERT_FLOAT_EQ(data[3], 40.0f);    // result[0,3] = min(100,40)
  ASSERT_FLOAT_EQ(data[4], 50.0f);    // result[0,4] = min(100,50)
  ASSERT_FLOAT_EQ(data[5], 100.0f);   // result[1,0] = 100 (no src)
  ASSERT_FLOAT_EQ(data[6], 20.0f);    // result[1,1] = min(100,20)
  ASSERT_FLOAT_EQ(data[7], 100.0f);   // result[1,2] = 100 (no src)
  ASSERT_FLOAT_EQ(data[8], 100.0f);   // result[1,3] = 100 (no src)
  ASSERT_FLOAT_EQ(data[9], 100.0f);   // result[1,4] = 100 (no src)
  ASSERT_FLOAT_EQ(data[10], 100.0f);  // result[2,0] = 100 (no src)
  ASSERT_FLOAT_EQ(data[11], 100.0f);  // result[2,1] = 100 (no src)
  ASSERT_FLOAT_EQ(data[12], 30.0f);   // result[2,2] = min(100,30)
  ASSERT_FLOAT_EQ(data[13], 100.0f);  // result[2,3] = 100 (no src)
  ASSERT_FLOAT_EQ(data[14], 100.0f);  // result[2,4] = 100 (no src)
}

TEST_F(ScatterReduceTest, ScatterReduceIncludeSelf) {
  at::Tensor self = at::full({3, 5}, 5.0f, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;

  // include_self=false: mapped positions use src only,
  // unmapped positions keep self original value.
  at::Tensor result = self.scatter_reduce(0, index, src, "sum", false);

  ASSERT_EQ(result.sizes()[0], 3);
  ASSERT_EQ(result.sizes()[1], 5);
  float* data = result.data_ptr<float>();
  // Mapped positions: src value only (self initial 5.0 ignored)
  ASSERT_FLOAT_EQ(data[0], 1.0f);   // result[0,0] = src[0,0]=1
  ASSERT_FLOAT_EQ(data[3], 4.0f);   // result[0,3] = src[0,3]=4
  ASSERT_FLOAT_EQ(data[4], 5.0f);   // result[0,4] = src[0,4]=5
  ASSERT_FLOAT_EQ(data[6], 2.0f);   // result[1,1] = src[0,1]=2
  ASSERT_FLOAT_EQ(data[12], 3.0f);  // result[2,2] = src[0,2]=3
  // Unmapped positions: keep self original value 5.0
  ASSERT_FLOAT_EQ(data[1], 5.0f);   // result[0,1]
  ASSERT_FLOAT_EQ(data[2], 5.0f);   // result[0,2]
  ASSERT_FLOAT_EQ(data[5], 5.0f);   // result[1,0]
  ASSERT_FLOAT_EQ(data[7], 5.0f);   // result[1,2]
  ASSERT_FLOAT_EQ(data[8], 5.0f);   // result[1,3]
  ASSERT_FLOAT_EQ(data[9], 5.0f);   // result[1,4]
  ASSERT_FLOAT_EQ(data[10], 5.0f);  // result[2,0]
  ASSERT_FLOAT_EQ(data[11], 5.0f);  // result[2,1]
  ASSERT_FLOAT_EQ(data[13], 5.0f);  // result[2,3]
  ASSERT_FLOAT_EQ(data[14], 5.0f);  // result[2,4]
}

TEST_F(ScatterReduceTest, ScatterReduceInvalidReduce) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  at::Tensor index = at::zeros({1, 1}, at::kLong);
  at::Tensor src = at::ones({1, 1}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce(0, index, src, "invalid_mode"),
               std::invalid_argument);
}

TEST_F(ScatterReduceTest, ScatterReduceNegativeIndex) {
  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = -1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::ones({2, 4}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce(1, index, src, "sum"), std::out_of_range);
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceNegativeIndex) {
  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = -1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::ones({2, 4}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce_(1, index, src, "sum"), std::out_of_range);
}

TEST_F(ScatterReduceTest, ScatterReduceIndexUpperBound) {
  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 4;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::ones({2, 4}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce(1, index, src, "sum"), std::out_of_range);
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceIndexUpperBound) {
  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 4;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::ones({2, 4}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce_(1, index, src, "sum"), std::out_of_range);
}

// Libtorch 2.12.1 C++ scatter_reduce accepts int32 index tensors.
TEST_F(ScatterReduceTest, ScatterReduceIntIndex) {
  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kInt);
  index.data_ptr<int>()[0] = 0;
  index.data_ptr<int>()[1] = 1;
  index.data_ptr<int>()[2] = 2;
  index.data_ptr<int>()[3] = 0;
  index.data_ptr<int>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "sum");

  float* data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[3], 4.0f);
  ASSERT_FLOAT_EQ(data[4], 5.0f);
  ASSERT_FLOAT_EQ(data[6], 2.0f);
  ASSERT_FLOAT_EQ(data[12], 3.0f);
}

// Libtorch 2.12.1 C++ scatter_reduce_ accepts int32 index tensors.
TEST_F(ScatterReduceTest, ScatterReduceInplaceIntIndex) {
  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kInt);
  index.data_ptr<int>()[0] = 0;
  index.data_ptr<int>()[1] = 1;
  index.data_ptr<int>()[2] = 2;
  index.data_ptr<int>()[3] = 0;
  index.data_ptr<int>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;

  self.scatter_reduce_(0, index, src, "sum");

  float* data = self.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 1.0f);
  ASSERT_FLOAT_EQ(data[3], 4.0f);
  ASSERT_FLOAT_EQ(data[4], 5.0f);
  ASSERT_FLOAT_EQ(data[6], 2.0f);
  ASSERT_FLOAT_EQ(data[12], 3.0f);
}

TEST_F(ScatterReduceTest, ScatterReduceFloatIndexThrows) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  at::Tensor index = at::zeros({1, 2}, at::kFloat);
  at::Tensor src = at::ones({1, 2}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce(0, index, src, "sum"),
               std::invalid_argument);
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceFloatIndexThrows) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  at::Tensor index = at::zeros({1, 2}, at::kFloat);
  at::Tensor src = at::ones({1, 2}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce_(0, index, src, "sum"),
               std::invalid_argument);
}

// Libtorch 2.12.1 C++ accepts empty floating index tensors.
TEST_F(ScatterReduceTest, ScatterReduceEmptyFloatIndex) {
  at::Tensor self = at::zeros({0, 2}, at::kFloat);
  at::Tensor index = at::empty({0, 2}, at::kFloat);
  at::Tensor src = at::empty({0, 2}, at::kFloat);

  at::Tensor result = self.scatter_reduce(0, index, src, "sum");

  ASSERT_EQ(result.sizes()[0], 0);
  ASSERT_EQ(result.sizes()[1], 2);
  ASSERT_EQ(result.numel(), 0);
}

TEST_F(ScatterReduceTest, ScatterReduceEmptyIndexRankMismatch) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  float* self_data = self.data_ptr<float>();
  self_data[0] = 1.0f;
  self_data[1] = 2.0f;
  self_data[2] = 3.0f;
  self_data[3] = 4.0f;
  at::Tensor index = at::empty({0}, at::kLong);
  at::Tensor src = at::empty({0}, at::kFloat);

  at::Tensor result = self.scatter_reduce(0, index, src, "sum");

  ASSERT_EQ(result.sizes()[0], 2);
  ASSERT_EQ(result.sizes()[1], 2);
  ASSERT_EQ(result.numel(), 4);
  float* result_data = result.data_ptr<float>();
  ASSERT_FLOAT_EQ(result_data[0], 1.0f);
  ASSERT_FLOAT_EQ(result_data[1], 2.0f);
  ASSERT_FLOAT_EQ(result_data[2], 3.0f);
  ASSERT_FLOAT_EQ(result_data[3], 4.0f);
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceEmptyIndexRankMismatch) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  float* self_data = self.data_ptr<float>();
  self_data[0] = 1.0f;
  self_data[1] = 2.0f;
  self_data[2] = 3.0f;
  self_data[3] = 4.0f;
  at::Tensor index = at::empty({0}, at::kLong);
  at::Tensor src = at::empty({0}, at::kFloat);

  self.scatter_reduce_(0, index, src, "sum");

  self_data = self.data_ptr<float>();
  ASSERT_FLOAT_EQ(self_data[0], 1.0f);
  ASSERT_FLOAT_EQ(self_data[1], 2.0f);
  ASSERT_FLOAT_EQ(self_data[2], 3.0f);
  ASSERT_FLOAT_EQ(self_data[3], 4.0f);
}

TEST_F(ScatterReduceTest, ScatterReduceEmptyIndexSrcDtypeMismatchThrows) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  at::Tensor index = at::empty({0}, at::kLong);
  at::Tensor src = at::empty({0}, at::kInt);

  EXPECT_THROW(self.scatter_reduce(0, index, src, "sum"), std::exception);
  EXPECT_THROW(self.scatter_reduce_(0, index, src, "sum"), std::exception);
}

TEST_F(ScatterReduceTest, ScatterReduceRankMismatchThrows) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  at::Tensor index = at::zeros({2}, at::kLong);
  at::Tensor src = at::ones({2, 2}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce(0, index, src, "sum"), std::exception);
}

TEST_F(ScatterReduceTest, ScatterReduceIndexLargerThanSrcThrows) {
  at::Tensor self = at::zeros({3, 2}, at::kFloat);
  at::Tensor index = at::zeros({3, 2}, at::kLong);
  at::Tensor src = at::ones({2, 2}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce(0, index, src, "sum"), std::exception);
}

TEST_F(ScatterReduceTest, ScatterReduceIndexLargerThanSelfThrows) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  at::Tensor index = at::zeros({1, 3}, at::kLong);
  at::Tensor src = at::ones({1, 3}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce(0, index, src, "sum"), std::exception);
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceShapeMismatchThrows) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  at::Tensor index = at::zeros({1, 3}, at::kLong);
  at::Tensor src = at::ones({1, 3}, at::kFloat);

  EXPECT_THROW(self.scatter_reduce_(0, index, src, "sum"), std::exception);
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceProd) {
  at::Tensor self = at::ones({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 2.0f;
  src.data_ptr<float>()[1] = 3.0f;
  src.data_ptr<float>()[2] = 4.0f;
  src.data_ptr<float>()[3] = 5.0f;
  src.data_ptr<float>()[4] = 6.0f;

  self.scatter_reduce_(0, index, src, "prod");

  float* data = self.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 2.0f);   // self[0,0] = 1*2
  ASSERT_FLOAT_EQ(data[3], 5.0f);   // self[0,3] = 1*5
  ASSERT_FLOAT_EQ(data[4], 6.0f);   // self[0,4] = 1*6
  ASSERT_FLOAT_EQ(data[6], 3.0f);   // self[1,1] = 1*3
  ASSERT_FLOAT_EQ(data[12], 4.0f);  // self[2,2] = 1*4
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceAmax) {
  at::Tensor self = at::zeros({2, 3}, at::kFloat);
  at::Tensor index = at::zeros({2, 3}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 0;
  index.data_ptr<int64_t>()[2] = 1;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 1;
  index.data_ptr<int64_t>()[5] = 1;
  at::Tensor src = at::full({2, 3}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 100.0f;
  src.data_ptr<float>()[4] = 200.0f;
  src.data_ptr<float>()[5] = 300.0f;

  self.scatter_reduce_(0, index, src, "amax");

  float* data = self.data_ptr<float>();
  ASSERT_FLOAT_EQ(data[0], 100.0f);  // self[0,0] = max(10, 100)
  ASSERT_FLOAT_EQ(data[1], 20.0f);   // self[0,1]
  ASSERT_FLOAT_EQ(data[4], 200.0f);  // self[1,1]
  ASSERT_FLOAT_EQ(data[5], 300.0f);  // self[1,2]
}

TEST_F(ScatterReduceTest, ScatterReduceNoIncludeSelfAmax) {
  // Use self=25.0f so some src values are below self (10, 20) and some above
  // (30, 40, 50). This ensures amax with include_self=false differs from true.
  at::Tensor self = at::full({3, 5}, 25.0f, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "amax", false);

  float* data = result.data_ptr<float>();
  // Mapped positions: src value only (self initial 25.0 ignored)
  ASSERT_FLOAT_EQ(data[0], 10.0f);   // result[0,0] = src[0,0]=10
  ASSERT_FLOAT_EQ(data[3], 40.0f);   // result[0,3] = src[0,3]=40
  ASSERT_FLOAT_EQ(data[4], 50.0f);   // result[0,4] = src[0,4]=50
  ASSERT_FLOAT_EQ(data[6], 20.0f);   // result[1,1] = src[0,1]=20
  ASSERT_FLOAT_EQ(data[12], 30.0f);  // result[2,2] = src[0,2]=30
  // Unmapped positions: keep self original value 25.0
  ASSERT_FLOAT_EQ(data[1], 25.0f);   // result[0,1]
  ASSERT_FLOAT_EQ(data[2], 25.0f);   // result[0,2]
  ASSERT_FLOAT_EQ(data[5], 25.0f);   // result[1,0]
  ASSERT_FLOAT_EQ(data[7], 25.0f);   // result[1,2]
  ASSERT_FLOAT_EQ(data[8], 25.0f);   // result[1,3]
  ASSERT_FLOAT_EQ(data[9], 25.0f);   // result[1,4]
  ASSERT_FLOAT_EQ(data[10], 25.0f);  // result[2,0]
  ASSERT_FLOAT_EQ(data[11], 25.0f);  // result[2,1]
  ASSERT_FLOAT_EQ(data[13], 25.0f);  // result[2,3]
  ASSERT_FLOAT_EQ(data[14], 25.0f);  // result[2,4]
}

TEST_F(ScatterReduceTest, ScatterReduceNoIncludeSelfAmin) {
  at::Tensor self = at::full({3, 5}, 5.0f, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "amin", false);

  float* data = result.data_ptr<float>();
  // Mapped positions: src value only (self initial 5.0 ignored)
  ASSERT_FLOAT_EQ(data[0], 10.0f);   // result[0,0] = src[0,0]=10
  ASSERT_FLOAT_EQ(data[3], 40.0f);   // result[0,3] = src[0,3]=40
  ASSERT_FLOAT_EQ(data[4], 50.0f);   // result[0,4] = src[0,4]=50
  ASSERT_FLOAT_EQ(data[6], 20.0f);   // result[1,1] = src[0,1]=20
  ASSERT_FLOAT_EQ(data[12], 30.0f);  // result[2,2] = src[0,2]=30
  // Unmapped positions: keep self original value 5.0
  ASSERT_FLOAT_EQ(data[1], 5.0f);   // result[0,1]
  ASSERT_FLOAT_EQ(data[2], 5.0f);   // result[0,2]
  ASSERT_FLOAT_EQ(data[5], 5.0f);   // result[1,0]
  ASSERT_FLOAT_EQ(data[7], 5.0f);   // result[1,2]
  ASSERT_FLOAT_EQ(data[8], 5.0f);   // result[1,3]
  ASSERT_FLOAT_EQ(data[9], 5.0f);   // result[1,4]
  ASSERT_FLOAT_EQ(data[10], 5.0f);  // result[2,0]
  ASSERT_FLOAT_EQ(data[11], 5.0f);  // result[2,1]
  ASSERT_FLOAT_EQ(data[13], 5.0f);  // result[2,3]
  ASSERT_FLOAT_EQ(data[14], 5.0f);  // result[2,4]
}

TEST_F(ScatterReduceTest, ScatterReduceNoIncludeSelfMean) {
  at::Tensor self = at::full({3, 5}, 5.0f, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "mean", false);

  float* data = result.data_ptr<float>();
  // Mapped positions: src value only (mean starts from 0, count=1)
  ASSERT_FLOAT_EQ(data[0], 10.0f);   // result[0,0] = src[0,0]=10
  ASSERT_FLOAT_EQ(data[3], 40.0f);   // result[0,3] = src[0,3]=40
  ASSERT_FLOAT_EQ(data[4], 50.0f);   // result[0,4] = src[0,4]=50
  ASSERT_FLOAT_EQ(data[6], 20.0f);   // result[1,1] = src[0,1]=20
  ASSERT_FLOAT_EQ(data[12], 30.0f);  // result[2,2] = src[0,2]=30
  // Unmapped positions: keep self original value 5.0
  ASSERT_FLOAT_EQ(data[1], 5.0f);   // result[0,1]
  ASSERT_FLOAT_EQ(data[2], 5.0f);   // result[0,2]
  ASSERT_FLOAT_EQ(data[5], 5.0f);   // result[1,0]
  ASSERT_FLOAT_EQ(data[7], 5.0f);   // result[1,2]
  ASSERT_FLOAT_EQ(data[8], 5.0f);   // result[1,3]
  ASSERT_FLOAT_EQ(data[9], 5.0f);   // result[1,4]
  ASSERT_FLOAT_EQ(data[10], 5.0f);  // result[2,0]
  ASSERT_FLOAT_EQ(data[11], 5.0f);  // result[2,1]
  ASSERT_FLOAT_EQ(data[13], 5.0f);  // result[2,3]
  ASSERT_FLOAT_EQ(data[14], 5.0f);  // result[2,4]
}

TEST_F(ScatterReduceTest, ScatterReduceNoIncludeSelfProd) {
  at::Tensor self = at::full({3, 5}, 5.0f, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 2.0f;
  src.data_ptr<float>()[1] = 3.0f;
  src.data_ptr<float>()[2] = 4.0f;
  src.data_ptr<float>()[3] = 5.0f;
  src.data_ptr<float>()[4] = 6.0f;

  at::Tensor result = self.scatter_reduce(0, index, src, "prod", false);

  float* data = result.data_ptr<float>();
  // Mapped positions: src value only (self initial 5.0 ignored)
  ASSERT_FLOAT_EQ(data[0], 2.0f);   // result[0,0] = src[0,0]=2
  ASSERT_FLOAT_EQ(data[3], 5.0f);   // result[0,3] = src[0,3]=5
  ASSERT_FLOAT_EQ(data[4], 6.0f);   // result[0,4] = src[0,4]=6
  ASSERT_FLOAT_EQ(data[6], 3.0f);   // result[1,1] = src[0,1]=3
  ASSERT_FLOAT_EQ(data[12], 4.0f);  // result[2,2] = src[0,2]=4
  // Unmapped positions: keep self original value 5.0
  ASSERT_FLOAT_EQ(data[1], 5.0f);   // result[0,1]
  ASSERT_FLOAT_EQ(data[2], 5.0f);   // result[0,2]
  ASSERT_FLOAT_EQ(data[5], 5.0f);   // result[1,0]
  ASSERT_FLOAT_EQ(data[7], 5.0f);   // result[1,2]
  ASSERT_FLOAT_EQ(data[8], 5.0f);   // result[1,3]
  ASSERT_FLOAT_EQ(data[9], 5.0f);   // result[1,4]
  ASSERT_FLOAT_EQ(data[10], 5.0f);  // result[2,0]
  ASSERT_FLOAT_EQ(data[11], 5.0f);  // result[2,1]
  ASSERT_FLOAT_EQ(data[13], 5.0f);  // result[2,3]
  ASSERT_FLOAT_EQ(data[14], 5.0f);  // result[2,4]
}

TEST_F(ScatterReduceTest, ScatterReduceDimOutOfRange) {
  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  at::Tensor index = at::zeros({1, 1}, at::kLong);
  at::Tensor src = at::ones({1, 1}, at::kFloat);

  // INT_MAX + 1 should trigger out_of_range
  EXPECT_THROW(
      self.scatter_reduce(static_cast<int64_t>(INT_MAX) + 1, index, src, "sum"),
      std::out_of_range);
  // INT_MIN - 1 should trigger out_of_range
  EXPECT_THROW(
      self.scatter_reduce(static_cast<int64_t>(INT_MIN) - 1, index, src, "sum"),
      std::out_of_range);
}
