//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/common/place.h"

__global__ void test(size_t* a, int size) {
  CUDA_KERNEL_LOOP(i, size) { a[i] *= 2; }
}

TEST(LoD, data) {
  paddle::framework::InitDevices();

  phi::LegacyLoD lod{{0, 1, 2}};
  lod.push_back({0, 2, 4, 5});
  lod.push_back(std::vector<size_t>({0, 1, 6, 8, 10, 11}));

  auto& v = lod[0];
  phi::MixVector<size_t> mix_vector_v(&v);
  phi::GPUPlace gpu(0);
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL(test,
                     dim3(1),
                     dim3(1),
                     0,
                     0,
                     mix_vector_v.CUDAMutableData(gpu),
                     v.size());
  hipDeviceSynchronize();
#else
  test<<<1, 1>>>(mix_vector_v.CUDAMutableData(gpu), v.size());
  cudaDeviceSynchronize();
#endif
  mix_vector_v.CopyToCPU();
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v[i], i * 2);
  }
}

TEST(DenseTensor, LoDInGPU) {
  paddle::framework::InitDevices();

  phi::DenseTensor lod_tensor;
  phi::GPUPlace place(0);

  phi::LegacyLoD src_lod;
  src_lod.push_back(std::vector<size_t>{0, 2, 4, 6, 8, 10, 12, 14});

  lod_tensor.Resize({14, 16});
  lod_tensor.mutable_data<float>(place);

  lod_tensor.set_lod(src_lod);
  EXPECT_EQ(lod_tensor.lod_element(0, 2).first, 4UL);
  EXPECT_EQ(lod_tensor.lod_element(0, 4).first, 8UL);

  auto lod = lod_tensor.lod();
  phi::MixVector<size_t> mix_vector(&(lod[0]));

#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL(test,
                     dim3(1),
                     dim3(8),
                     0,
                     0,
                     mix_vector.CUDAMutableData(place),
                     lod[0].size());
  hipDeviceSynchronize();
#else
  test<<<1, 8>>>(mix_vector.CUDAMutableData(place), lod[0].size());
  cudaDeviceSynchronize();
#endif
  mix_vector.CopyToCPU();

  for (size_t i = 0; i < src_lod[0].size(); ++i) {
    EXPECT_EQ(lod[0].data()[i], src_lod[0].data()[i] * 2);
  }
}
