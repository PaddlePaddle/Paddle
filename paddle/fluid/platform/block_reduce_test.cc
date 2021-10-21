/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <time.h>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/block_reduce.cuh"
#include "paddle/fluid/platform/dynload/"

namespace paddle {
namespace platform {

template <typename T, int N, int x, int y, int row, int col>
void CallBlockReduce() {
  void *input = nullptr, *output = nullptr;
  cudaMalloc(&input, sizeof(T) * row * col);
  cudaMalloc(&output, sizeof(T) * col);

  float *data = new float[row * col];
  float *result = new float[col];

  for (int idx = 0; idx < row * col; ++idx) {
    data[idx] = rand_t(time(NULL)) / 1000.0f;
  }

  cudaMemcpy(input, data, sizeof(float) * row * col, cudaMemcpyHostToDevice);

  dim3 dg(col / (N * x));
  dim3 db(x, y);
  reduce_block<T, N, x, y, T, row, col><<<dg, db, 0, 0>>>(input, row, col,
                                                          output);
  cudaMemcpy(result, output, sizeof(float) * col, cudaMemcpyDeviceToHost);

  cudaFree(input);
  cudaFree(output);
}

TEST(block_reduce, call_block_reduce) {}

}  // namespace platform
}  // namespace paddle
