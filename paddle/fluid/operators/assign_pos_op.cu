/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The file has been adapted from the two files:
     https://github.com/laekov/fastmoe/blob/master/cuda/local_exchange.cu
     https://github.com/laekov/fastmoe/blob/master/cuda/local_exchange.cuh
     Git commit hash: 295a615aacce7e54a37e7935274ba15e901c78e4
We retain the following license from the original files:
         Copyright 2021, Jiaao He
   Licensed under the Apache License, Version 2.0 (the "License").
*/

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/assign_pos_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"

DECLARE_bool(avoid_op_randomness);

namespace paddle {
namespace operators {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void AssignPos(T* cum_count, const T* numbers, T* out,
                          int64_t limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    int number_idx = numbers[i];
    if (number_idx > -1) {
      int p = platform::CudaAtomicAdd(cum_count + number_idx, -1);
      out[p - 1] = i;
    }
  }
}

template <typename T>
class AssignPosCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // assign pos decides which tokens should be fetched belong to specially
    // counter orderingly.
    auto cum_count = context.Input<LoDTensor>(
        "cum_count");  // (counter number) int32 | int64
    auto numbers =
        context.Input<LoDTensor>("X");  // (batch_size * seq_len, topk) int32
    auto eff_num_len =
        context.Input<LoDTensor>("eff_num_len");  // (sum(cum_count))
    auto out = context.Output<LoDTensor>("Out");  // (cum_count) value ranges
                                                  // from 0 to batch_size *
                                                  // seq_len * topk
    auto place = context.GetPlace();
    auto numel = numbers->numel();
    T* cum_data = const_cast<T*>(cum_count->data<T>());
    auto cum_size = cum_count->numel();

    framework::Tensor cpu_eff_num_len;
    int64_t cpu_eff_num_len_data = 0;
    if (platform::is_cpu_place(eff_num_len->place())) {
      cpu_eff_num_len_data = eff_num_len->data<T>()[0];
    } else {
      framework::TensorCopySync(*eff_num_len, platform::CPUPlace(),
                                &cpu_eff_num_len);
      cpu_eff_num_len_data = cpu_eff_num_len.data<T>()[0];
    }
    const auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    framework::DDim out_dims = phi::make_ddim({cpu_eff_num_len_data});
    auto out_data = out->mutable_data<T>(out_dims, place);

    const T* num_data = numbers->data<T>();

    int blocks = NumBlocks(numel);
    int threads = kNumCUDAThreads;

    AssignPos<T><<<blocks, threads, 0, dev_ctx.stream()>>>(cum_data, num_data,
                                                           out_data, numel);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(assign_pos, ops::AssignPosCUDAKernel<int64_t>);
