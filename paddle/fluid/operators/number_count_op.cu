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
//
// The file has been adapted from the two files:
//     https://github.com/laekov/fastmoe/blob/master/cuda/local_exchange.cu
//     https://github.com/laekov/fastmoe/blob/master/cuda/local_exchange.cuh
//     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
// We retain the following license from the original files:
//     Copyright 2021, Jiaao He. All rights reserved.
//  Licensed under the Apache License, Version 2.0 (the "License").

#include "paddle/fluid/operators/number_count_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

#define CEIL(_x_, _y_) (((_x_)-1) / (_y_) + 1)
#define PERTHREAD_EXPERTS 256
#define WARP_SIZE 32

const int CUDA_NUM_THREADS = 512;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void initialize_zero_kernel(T* data, const int length) {
  CUDA_KERNEL_LOOP(idx, length) { data[idx] = static_cast<T>(0); }
}

template <typename T>
__global__ void NumberCount(const T* numbers,
                            T* number_count,
                            int64_t batch_size,
                            int upper_range) {
  int res_tmp[PERTHREAD_EXPERTS] = {0};
  int expert_min = blockIdx.x * PERTHREAD_EXPERTS;
  int expert_max = expert_min + PERTHREAD_EXPERTS;
  if (expert_max > upper_range) {
    expert_max = upper_range;
  }
  for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
    T idx = numbers[i];
    if (idx == -1) {
      continue;
    }
    if (idx < expert_min || idx >= expert_max) {
      continue;
    }
    res_tmp[idx - expert_min] += 1;
  }
  for (int i = expert_min; i < expert_max; ++i) {
    int x = res_tmp[i - expert_min];
#pragma unroll
    for (int j = 1; j < WARP_SIZE; j <<= 1) {
#ifdef __HIPCC__
      x = x + __shfl_down(x, j);
#else
      x = x + __shfl_down_sync(-1u, x, j);
#endif
    }
    if (threadIdx.x % WARP_SIZE == 0) {
      phi::CudaAtomicAdd(number_count + i, x);
    }
  }
}

template <typename T>
class NumberCountOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto numbers = context.Input<phi::DenseTensor>("numbers");
    auto upper_range = context.Attr<int>("upper_range");
    auto number_count = context.Output<phi::DenseTensor>("Out");

    int64_t batch_size = numbers->numel();
    auto place = context.GetPlace();
    const auto& dev_ctx = context.template device_context<phi::GPUContext>();

    framework::DDim out_dims = phi::make_ddim({upper_range});
    auto out_data = number_count->mutable_data<T>(out_dims, place);
    const T* gate_data = numbers->data<T>();

    initialize_zero_kernel<T>
        <<<GET_BLOCKS(upper_range), CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
            out_data, upper_range);

    NumberCount<T>
        <<<CEIL(upper_range, PERTHREAD_EXPERTS), 256, 0, dev_ctx.stream()>>>(
            gate_data, out_data, batch_size, upper_range);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(number_count, ops::NumberCountOpCUDAKernel<int64_t>);
