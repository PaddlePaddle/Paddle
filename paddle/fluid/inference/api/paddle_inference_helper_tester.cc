// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "paddle/fluid/inference/api/paddle_inference_helper.h"

namespace paddle {
namespace helper {

TEST(Timer, Basic) {
  Timer timer;
  timer.tic();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_GE(timer.toc(), 1000);  // ms
}

TEST(to_string, Basic) {
  std::vector<int> x({0, 1, 2, 3});
  auto str = to_string(x, ',');
  ASSERT_EQ(str, "0,1,2,3");
}

TEST(TensorSniffer, Basic) {
  PaddleTensor x;
  x.dtype = PaddleDType::FLOAT32;
  x.lod.emplace_back(std::vector<size_t>({0, 1, 2}));
  x.name = "x";
  x.shape.assign({10, 3});

  TensorSniffer sniffer(x);

  EXPECT_EQ(sniffer.dtype(), "float32");
  EXPECT_EQ(sniffer.shape(), "[10 3]");
  EXPECT_EQ(sniffer.lod(), "[[0 1 2],]");
}

}  // namespace helper
}  // namespace paddle
