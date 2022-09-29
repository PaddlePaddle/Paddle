// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/save_load_util.h"

#include <stdlib.h>
#include <time.h>

#include "gtest/gtest.h"

namespace paddle {
namespace framework {
TEST(test_save_load_util, test_save_load) {
  srand(time(NULL));
  auto cpu_place = platform::CPUPlace();
  phi::DenseTensor tensor1;
  tensor1.Resize({1000, 1000});
  auto src_data_1 = tensor1.mutable_data<float>(cpu_place);
  phi::DenseTensor tensor2;
  tensor2.Resize({5000, 1000});
  auto src_data_2 = tensor2.mutable_data<float>(cpu_place);

  for (int64_t i = 0; i < tensor1.numel(); ++i) {
    float temp = (rand() % 10000) * 1.0 / 50000 - 1.0;  // NOLINT

    src_data_1[i] = temp;
  }

  for (int64_t i = 0; i < tensor2.numel(); ++i) {
    float temp = (rand() % 10000) * 1.0 / 50000 - 1.0;  // NOLINT

    src_data_2[i] = temp;
  }

  std::map<std::string, phi::DenseTensor*> map_tensor;
  map_tensor["t1"] = &tensor1;
  map_tensor["t2"] = &tensor2;

  SaveTensorToDisk("test_1", map_tensor);

  std::map<std::string, std::shared_ptr<phi::DenseTensor>> load_map_tensor;

  LoadTensorFromDisk("test_1", &load_map_tensor);

  ASSERT_TRUE(load_map_tensor.find("t1") != load_map_tensor.end());
  ASSERT_TRUE(load_map_tensor.find("t2") != load_map_tensor.end());

  auto new_tensor_1 = load_map_tensor["t1"];
  auto new_tensor_2 = load_map_tensor["t2"];

  float* ptr_1 = tensor1.data<float>();
  float* ptr_1_new = new_tensor_1->data<float>();

  for (int64_t i = 0; i < tensor1.numel(); ++i) {
    ASSERT_EQ(ptr_1[i], ptr_1_new[i]);
  }

  float* ptr_2 = tensor2.data<float>();
  float* ptr_2_new = new_tensor_2->data<float>();

  for (int64_t i = 0; i < tensor2.numel(); ++i) {
    ASSERT_EQ(ptr_2[i], ptr_2_new[i]);
  }
}
}  // namespace framework
}  // namespace paddle
