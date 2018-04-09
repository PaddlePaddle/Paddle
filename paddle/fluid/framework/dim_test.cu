//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "hip/hip_runtime.h"
#include <thrust/device_vector.h>
#include <sstream>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/dim.h"

__global__ void test(paddle::framework::Dim<2>* o) {
  o[0] = paddle::framework::make_dim(5, 6);
}

__global__ void dyn_idx_gpu(int64_t* o) {
  auto d = paddle::framework::make_dim(5, 6);
  o[0] = d[1];
}

TEST(Dim, Equality) {
  // construct a Dim on the CPU
  auto a = paddle::framework::make_dim(3, 4);
  EXPECT_EQ(paddle::framework::get<0>(a), 3);
  EXPECT_EQ(paddle::framework::get<1>(a), 4);

  // construct a Dim on the GPU
  thrust::device_vector<paddle::framework::Dim<2>> t(2);
  hipLaunchKernelGGL((test), dim3(1), dim3(1), 0, 0, thrust::raw_pointer_cast(t.data()));
  a = t[0];
  EXPECT_EQ(paddle::framework::get<0>(a), 5);
  EXPECT_EQ(paddle::framework::get<1>(a), 6);

  // linearization
  auto b = paddle::framework::make_dim(7, 8);
  EXPECT_EQ(paddle::framework::linearize(a, b), 83);

  // product
  EXPECT_EQ(paddle::framework::product(a), 30);

  // mutate a Dim
  paddle::framework::get<1>(b) = 10;
  EXPECT_EQ(paddle::framework::get<0>(b), 7);
  EXPECT_EQ(paddle::framework::get<1>(b), 10);

  // dynamic access
  paddle::framework::get(b, 0) = 8;
  b[1] = 11;
  EXPECT_EQ(paddle::framework::get<0>(b), 8);
  EXPECT_EQ(paddle::framework::get<1>(b), 11);
  EXPECT_EQ(paddle::framework::get(b, 0), 8);
  EXPECT_EQ(b[1], 11);

  // dynamic access on GPU
  thrust::device_vector<int64_t> r(1);
  hipLaunchKernelGGL((dyn_idx_gpu), dim3(1), dim3(1), 0, 0, thrust::raw_pointer_cast(r.data()));
  int64_t res = r[0];
  EXPECT_EQ(res, 6);

  // ex_prefix_mul
  paddle::framework::Dim<3> c =
      paddle::framework::ex_prefix_mul(paddle::framework::Dim<3>(3, 4, 5));
  EXPECT_EQ(paddle::framework::get<0>(c), 1);
  EXPECT_EQ(paddle::framework::get<1>(c), 3);
  EXPECT_EQ(paddle::framework::get<2>(c), 12);

  // generate from an index
  auto size = paddle::framework::make_dim(4, 5, 2);
  c = paddle::framework::Dim<3>(14, size);
  EXPECT_EQ(paddle::framework::get<0>(c), 2);
  EXPECT_EQ(paddle::framework::get<1>(c), 3);
  EXPECT_EQ(paddle::framework::get<2>(c), 0);
  c = paddle::framework::Dim<3>(25, size);
  EXPECT_EQ(paddle::framework::get<0>(c), 1);
  EXPECT_EQ(paddle::framework::get<1>(c), 1);
  EXPECT_EQ(paddle::framework::get<2>(c), 1);
}

TEST(Dim, Bool) {
  auto a = paddle::framework::make_dim(3, 4);
  auto b = paddle::framework::make_dim(5, 6);
  auto c = paddle::framework::make_dim(3, 4);

  // in_bounds check
  EXPECT_TRUE(paddle::framework::contained(a, b));
  EXPECT_FALSE(paddle::framework::contained(b, a));

  // comparison
  EXPECT_TRUE(a == a);
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a == c);
}

TEST(Dim, Print) {
  {
    std::stringstream ss;
    auto a = paddle::framework::make_dim(2, 3);
    ss << a;
    EXPECT_EQ(ss.str(), "2, 3");
  }
  {
    std::stringstream ss;
    ss << paddle::framework::make_dim(8);
    EXPECT_EQ(ss.str(), "8");
  }
}
