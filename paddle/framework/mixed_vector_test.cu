/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#include <cuda.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

#include "paddle/framework/init.h"
#include "paddle/framework/mixed_vector.h"

using namespace paddle::framework;
using namespace paddle::platform;
using namespace paddle::memory;

template <typename T>
__global__ void test(T* data, int size) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    data[i] *= 2;
  }
}

TEST(Vector, Normal) {
  // fill the device context pool.
  InitDevices();

  Vector<size_t> vec({1, 2, 3});
  size_t* ptr = vec.data();
  for (size_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], *(ptr + i));
  }

  vec.clear();
  vec.CopyFromCUDA();

  std::vector<size_t> v = {1, 2, 3};
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_EQ(v[i], vec[i]);
  }
}

TEST(Vector, MultipleCopy) {
  InitDevices();
  Vector<size_t> vec({1, 2, 3});
  CUDAPlace place(0);
  vec.mutable_data(place);
  auto vec2 = Vector<size_t>(vec);
  {
    const size_t* ptr = vec2.data(CPUPlace());
    for (size_t i = 0; i < vec2.size(); ++i) {
      EXPECT_EQ(*(ptr + i), vec[i]);
    }
  }
  test<size_t><<<3, 3>>>(vec2.mutable_data(place), vec2.size());
  vec2.CopyFromCUDA();
  {
    const size_t* ptr = vec2.data(CPUPlace());
    for (size_t i = 0; i < vec2.size(); ++i) {
      EXPECT_EQ(*(ptr + i), vec[i] * 2);
    }
  }
}
