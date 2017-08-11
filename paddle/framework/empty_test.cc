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

#include <gtest/gtest.h>
#include <string>
#include "paddle/framework/tensor.h"

TEST(Empty, Dims) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  Tensor tt;
  tt.Resize(make_ddim({0, 3, 4}));
  DDim dims = tt.dims();
  ASSERT_EQ(arity(dims), 3);
  EXPECT_EQ(0, dims[0]);
  EXPECT_EQ(3, dims[1]);
  EXPECT_EQ(4, dims[2]);
}

TEST(Empty, MutableData) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  {
    Tensor src_tensor;
    float* p1 = nullptr;
    // initialization
    p1 = src_tensor.mutable_data<float>(make_ddim({0, 2, 3}), CPUPlace());
    EXPECT_NE(p1, nullptr);
  }

#ifndef PADDLE_ONLY_CPU
  {
    Tensor src_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = src_tensor.mutable_data<float>(make_ddim({0, 2, 3}), GPUPlace());
    EXPECT_NE(p1, nullptr);
    // set src_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = src_tensor.mutable_data<float>(make_ddim({0, 4}), GPUPlace());
    EXPECT_NE(p2, nullptr);
    // EXPECT_NE(p1, p2);
  }
#endif
}
