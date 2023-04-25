// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/extension.h"

TEST(PaddleLib, CPU) {
  int data[] = {1, 2, 3, 4};
  auto tensor = paddle::from_blob(data, {2, 2}, phi::DataType::INT32);

  ASSERT_EQ(tensor.numel(), 4);
  ASSERT_EQ(tensor.dtype(), phi::DataType::INT32);
  ASSERT_EQ(tensor.is_cpu(), true);

  auto* tensor_data = tensor.template data<int>();
  ASSERT_EQ(data, tensor_data);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
