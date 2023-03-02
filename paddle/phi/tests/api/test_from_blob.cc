/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/from_blob.h"

namespace paddle {
namespace tests {

TEST(from_blob, FLOAT32) {
  // 1. create data
  float data[] = {1, 2, 3, 4, 5, 6};

  // 2. test API
  auto test_tesnor = experimental::from_blob(data, {2, 3}, phi::DataType::FLOAT32, phi::CPUPlace());

  // 3. check result
  // 3.1 check tensor attributes
  ASSERT_EQ(test_tesnor.dims().size(), 2);
  ASSERT_EQ(test_tesnor.dims()[0], 2);
  ASSERT_EQ(test_tesnor.dims()[1], 3);
  ASSERT_EQ(test_tesnor.numel(), 6);
  ASSERT_EQ(test_tesnor.is_cpu(), true);
  ASSERT_EQ(test_tesnor.dtype(), phi::DataType::FLOAT32);
  ASSERT_EQ(test_tesnor.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(test_tesnor.initialized(), true);
  ASSERT_EQ(test_tesnor.is_dense_tensor(), true);

  // 3.2 check tensor values
  auto* test_tensor_data = test_tesnor.template data<float>();
  for (int32_t i = 0; i < 6; i++) {
    ASSERT_EQ(test_tensor_data[i], static_cast<float>(i + 1));
  }

  // 3.3 check whether memory is shared
  ASSERT_EQ(data, test_tensor_data);

  // 3.4 test other API
  auto test_tensor_pow = pow(test_tesnor, 2);
  auto* test_tensor_pow_data = test_tensor_pow.template data<float>();
  for (int32_t i = 0; i < 6; i++) {
    ASSERT_EQ(test_tensor_pow_data[i], static_cast<float>(std::pow(i + 1, 2)));
  }
}

TEST(from_blob, INT32) {
  // 1. create data
  int32_t data[] = {4, 3, 2, 1};

  // 2. test API
  auto test_tesnor = experimental::from_blob(data, {1, 2, 2}, phi::DataType::INT32, phi::CPUPlace());

  // 3. check result
  // 3.1 check tensor attributes
  ASSERT_EQ(test_tesnor.dims().size(), 3);
  ASSERT_EQ(test_tesnor.dims()[0], 1);
  ASSERT_EQ(test_tesnor.dims()[1], 2);
  ASSERT_EQ(test_tesnor.dims()[2], 2);
  ASSERT_EQ(test_tesnor.numel(), 4);
  ASSERT_EQ(test_tesnor.is_cpu(), true);
  ASSERT_EQ(test_tesnor.dtype(), phi::DataType::INT32);
  ASSERT_EQ(test_tesnor.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(test_tesnor.initialized(), true);
  ASSERT_EQ(test_tesnor.is_dense_tensor(), true);

  // 3.2 check tensor values
  auto* test_tensor_data = test_tesnor.template data<int32_t>();
  for (int32_t i = 0; i < 4; i++) {
    ASSERT_EQ(test_tensor_data[i], 4 - i);
  }

  // 3.3 check whether memory is shared
  ASSERT_EQ(data, test_tensor_data);

  // 3.4 test other API
  auto test_tensor_pow = pow(test_tesnor, 2);
  auto* test_tensor_pow_data = test_tensor_pow.template data<int32_t>();
  for (int32_t i = 0; i < 4; i++) {
    ASSERT_EQ(test_tensor_pow_data[i], static_cast<int32_t>(std::pow(4 - i, 2)));
  }
}

}  // namespace tests
}  // namespace paddle