// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/autotune/cache.h"
#include <gtest/gtest.h>
#include <functional>
#include "glog/logging.h"

void Algo() { VLOG(3) << "algo test"; }

TEST(AlgosCache, AlgosCache) {
  phi::AlgorithmsCache<std::function<void()>> cache;
  std::vector<int> x_shape = {4, 224, 224, 3};
  std::vector<int> w_shape = {32, 3, 3, 3};
  std::vector<int> paddings = {0, 0};
  std::vector<int> strides = {2, 2};
  std::vector<int> dilations = {1, 1};
  int algorithmFlags = 2;
  int cudnn_dtype = 0;

  auto key = cache.GetKey(x_shape,
                          w_shape,
                          paddings,
                          strides,
                          dilations,
                          algorithmFlags,
                          cudnn_dtype);
  cache.Set(key, Algo);
  EXPECT_EQ(cache.Find(key), true);
  auto algo = cache.Get(key);
  algo();

  x_shape = {4, 128, 128, 3};
  key = cache.GetKey(x_shape,
                     w_shape,
                     paddings,
                     strides,
                     dilations,
                     algorithmFlags,
                     cudnn_dtype);
  EXPECT_EQ(cache.Find(key), false);
}
