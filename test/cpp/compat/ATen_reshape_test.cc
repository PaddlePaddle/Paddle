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
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/ops/tensor.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

TEST(ATenReshapeTest, BasicReshape) {
  // create a 2x3 tensor
  auto tensor = torch::arange(6, torch::kFloat32).view({2, 3});

  // use torch::reshape to reshape it to 3x2
  auto reshaped = torch::reshape(tensor, {3, 2});

  // verify the shape
  EXPECT_EQ(reshaped.size(0), 3);
  EXPECT_EQ(reshaped.size(1), 2);

  // verify the data content remains the same
  EXPECT_FLOAT_EQ(reshaped[0][0].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(reshaped[0][1].item<float>(), 1.0f);
  EXPECT_FLOAT_EQ(reshaped[1][0].item<float>(), 2.0f);
  EXPECT_FLOAT_EQ(reshaped[1][1].item<float>(), 3.0f);
  EXPECT_FLOAT_EQ(reshaped[2][0].item<float>(), 4.0f);
  EXPECT_FLOAT_EQ(reshaped[2][1].item<float>(), 5.0f);

  // verify the total number of elements remains the same
  EXPECT_EQ(tensor.numel(), reshaped.numel());
}
