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

#include "paddle/framework/tensor.h"
#include <gtest/gtest.h>
#include <string>

TEST(Tensor, ASSERT) {
  paddle::framework::Tensor cpu_tensor;

  bool caught = false;
  try {
    const double* p __attribute__((unused)) = cpu_tensor.data<double>();
  } catch (paddle::framework::EnforceNotMet err) {
    caught = true;
    std::string msg = "Tensor::data must be called after Tensor::mutable_data";
    const char* what = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(what[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);
}

/* mutable_data() is not tested at present
   because Memory::Alloc() and Memory::Free() have not been ready.

TEST(Tensor, MutableData) {
  using namespace paddle::framework;
  using namespace paddle::platform;
  {
    Tensor cpu_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = cpu_tensor.mutable_data<float>(make_ddim({1, 2, 3}), CPUPlace());
    EXPECT_NE(p1, nullptr);
    // set cpu_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = cpu_tensor.mutable_data<float>(make_ddim({3, 4}));
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    // set cpu_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = cpu_tensor.mutable_data<float>(make_ddim({2, 2, 3}));
    EXPECT_EQ(p1, p2);
    // set cpu_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = cpu_tensor.mutable_data<float>(make_ddim({2, 2}));
    EXPECT_EQ(p1, p2);
  }

  {
    Tensor gpu_tensor;
    float* p1 = nullptr;
    float* p2 = nullptr;
    // initialization
    p1 = gpu_tensor.mutable_data<float>(make_ddim({1, 2, 3}), GPUPlace());
    EXPECT_NE(p1, nullptr);
    // set gpu_tensor a new dim with large size
    // momery is supposed to be re-allocated
    p2 = gpu_tensor.mutable_data<float>(make_ddim({3, 4}));
    EXPECT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);
    // set gpu_tensor a new dim with same size
    // momery block is supposed to be unchanged
    p1 = gpu_tensor.mutable_data<float>(make_ddim({2, 2, 3}));
    EXPECT_EQ(p1, p2);
    // set gpu_tensor a new dim with smaller size
    // momery block is supposed to be unchanged
    p2 = gpu_tensor.mutable_data<float>(make_ddim({2, 2}));
    EXPECT_EQ(p1, p2);
  }
}
*/