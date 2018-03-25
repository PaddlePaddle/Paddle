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

#include "paddle/fluid/operators/sequence_softmax_op.h"

namespace paddle {
namespace operators {
#define FLT_MAX __FLT_MAX__

template <typename T>
__global__ void sequence_softmax(const T* x, const size_t num_classes,
                                 const size_t* lod, const size_t lod_size,
                                 T* out) {
  extern __shared__ T mem[];

  int bid = blockIdx.x;
  if (bid >= lod_size) return;
  int start_pos = static_cast<int>(lod[bid]);
  int end_pos = static_cast<int>(lod[bid + 1]);
  T* shm = &mem[start_pos];
  // get max logits
  for (int tid = threadIdx.x; tid < (end_pos - start_pos); tid += blockIdx.x) {
    T max_logits = static_cast<int>(FLT_MAX);
    int offset = (start_pos + tid) * num_classes;
    for (int i = 0; i < num_classes; ++i) {
      if (x[offset + i] > max_logits) {
        max_logits = ValueClip<T>(x[offset + i]);
      }
    }
    shm[tid] = max_logits;
  }
  __syncthreads();

  // compute softmax sum
  for (int tid = threadIdx.x; tid < (end_pos - start_pos); tid += blockIdx.x) {
    int offset = (start_pos + tid) * num_classes;
    T softmax_sum = static_cast<T>(0);
    for (int i = 0; i < num_classes; ++i) {
      out[offset + i] = exp(x[offset + i] - shm[tid]);
      softmax_sum += out[offset + i];
    }
    shm[tid] = 1.0 / softmax_sum;
  }
  __syncthreads();

  // generate per class softmax prob.
  for (int tid = threadIdx.x; tid < (end_pos - start_pos); tid += blockIdx.x) {
    int offset = (start_pos + tid) * num_classes;
    for (int i = 0; i < num_classes; ++i) {
      out[offset + i] = shm[tid] * out[offset + i]
    }
  }
}

template <typename T>
struct SequenceSoftmaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx, const LoDTensor& x,
                  LoDTensor* out) {
    const size_t level = x.lod().size() - 1;
    auto lod = x.lod()[level];
    dim3 threads(1024, 1);
    dim3 block(lod.size(), 1);
    sequence_softmax<
        T><<<block, threads, framework::product(x.dims()) * sizeof(T),
             ctx.stream()>>>(x.data<T>(), lod.CUDAData(ctx.GetPlace()),
                             lod.size(), out->mutable_data<T>(ctx.GetPlace()));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sequence_softmax,
    ops::SequenceSoftmaxKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SequenceSoftmaxKernel<paddle::platform::CUDADeviceContext, double>)
REGISTER_OP_CUDA_KERNEL(
    sequence_softmax_grad,
    ops::SequenceSoftmaxGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SequenceSoftmaxGradKernel<paddle::platform::CUDADeviceContext,
                                   double>);
