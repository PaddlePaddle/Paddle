// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/fluid/framework/tvm/tvm_runtime.h"

namespace paddle {
namespace framework {
namespace tvm {
namespace runtime {

template <typename T>
static __global__ void SetSequence(T *x, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    x[idx] = static_cast<T>(idx);
  }
}

template <typename T>
static __global__ void SetConstant(T *x, T val, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    x[idx] = val;
  }
}

template <typename T>
void TestMain() {
  platform::CUDAPlace place(0);
  auto f = GetFuncFromLib("test_addone_dll.so", "addone");

  Tensor x, y;
  int64_t n = (1L << 20);

  x.Resize({n});
  auto *x_data = x.mutable_data<T>(place);

  int threads = 512;
  int grids = (n + threads - 1) / threads;
  SetSequence<<<grids, threads>>>(x_data, n);

  y.Resize({n});
  auto *y_data = y.mutable_data<T>(place);
  SetConstant<<<grids, threads>>>(y_data, static_cast<T>(-1), n);

  cudaStream_t stream;
  PADDLE_ENFORCE(cudaStreamCreate(&stream));
  SetStream(place.device, stream);
  DLPackTensor tvm_x(x), tvm_y(y);
  PADDLE_ENFORCE(cudaStreamQuery(stream) == cudaSuccess);
  f(tvm_x, tvm_y);
  PADDLE_ENFORCE(cudaStreamQuery(stream) == cudaErrorNotReady);

  PADDLE_ENFORCE(cudaStreamSynchronize(stream));

  std::vector<T> ret(n, 0);
  cudaMemcpy(ret.data(), y_data, n * sizeof(T), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < ret.size(); ++i) {
    PADDLE_ENFORCE(ret[i], i + 1);
  }

  PADDLE_ENFORCE(cudaStreamDestroy(stream));
}

TEST(tvm_runtime, FP32) { TestMain<float>(); }

}  // namespace runtime
}  // namespace tvm
}  // namespace framework
}  // namespace paddle
