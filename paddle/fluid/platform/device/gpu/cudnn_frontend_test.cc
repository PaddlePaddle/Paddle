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

#include "paddle/fluid/platform/device/gpu/cuda/cudnn_frontend.h"

#include <gtest/gtest.h>
#include <iostream>

namespace paddle {
namespace platform {

TEST(CudnnFrontendTest, TensorCreation) {
  // Consider creation of a 2d Tensor
  // n,c,h,w as 4,32,32,32
  std::cout << "Tensor creation comparison" << std::endl;
  std::array<int64_t, 4> tensor_dim = {4, 32, 32, 32};
  std::array<int64_t, 4> tensor_str = {32768, 1024, 32, 1};  // NCHW format
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  int64_t alignment = sizeof(float);
  int64_t id = 0xD0D0CACA;  // Some magic number

  try {
    auto tensor = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_dim.size(), tensor_dim.data())
                      .setStrides(tensor_str.size(), tensor_str.data())
                      .setId(id)
                      .setAlignment(alignment)
                      .setDataType(data_type)
                      .build();
  } catch (cudnn_frontend::cudnnException &e) {
    std::cout << "Exception in tensor creation " << e.what() << std::endl;
  }
  std::cout << "Finished tensor creation." << std::endl;
}

}  // namespace platform
}  // namespace paddle
