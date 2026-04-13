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

#if defined(PADDLE_WITH_CUDA)

#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAStream.h>

#include "gtest/gtest.h"

namespace {

__global__ void WriteScalarKernel(int* dst, int value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    dst[0] = value;
  }
}

}  // namespace

TEST(ATenPinMemoryKernelTest, KernelCanWritePinnedTensorDirectly) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto tensor =
      at::empty({1}, at::TensorOptions().dtype(at::kInt).pinned_memory(true));

  ASSERT_TRUE(tensor.is_pinned());
  ASSERT_FALSE(tensor.is_cuda());

  tensor._PD_GetInner().data<int>()[0] = 0;
  WriteScalarKernel<<<1, 1, 0, stream>>>(tensor.data_ptr<int>(), 123);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(tensor._PD_GetInner().data<int>()[0], 123);
}

#endif  // PADDLE_WITH_CUDA
