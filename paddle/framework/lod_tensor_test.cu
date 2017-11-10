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

#include <cuda.h>
#include <cuda_runtime.h>
#include "paddle/framework/lod_tensor.h"
#include "paddle/platform/assert.h"

#include <gtest/gtest.h>

__global__ void test(size_t* a, int size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    a[i] *= 2;
  }
}

TEST(LoDTensor, LoDInGPU) {
  paddle::framework::LoDTensor lod_tensor;
  paddle::platform::GPUPlace place(0);

  paddle::framework::LoD src_lod;
  src_lod.push_back(std::vector<size_t>{0, 2, 4, 6, 8, 10, 12, 14});

  lod_tensor.Resize({14, 16});
  lod_tensor.mutable_data<float>(place);

  lod_tensor.set_lod(src_lod);
  EXPECT_EQ(lod_tensor.lod_element(0, 2).first, 4UL);
  EXPECT_EQ(lod_tensor.lod_element(0, 4).first, 8UL);

  auto lod = lod_tensor.lod();

  test<<<1, 8>>>(lod[0].data(), lod[0].size());
  cudaDeviceSynchronize();

  for (size_t i = 0; i < src_lod[0].size(); ++i) {
    EXPECT_EQ(lod[0].data()[i], src_lod[0].data()[i] * 2);
  }
}
