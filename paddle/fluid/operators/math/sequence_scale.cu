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

#include "paddle/fluid/operators/math/sequence_scale.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {
namespace math {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, int BlockSize>
__global__ void SequenceScaleKernel(T* seq, size_t* lod, const T* scales,
                                    const size_t seq_width) {
  for (int i = threadIdx.x;
       i < (lod[blockIdx.x + 1] - lod[blockIdx.x]) * seq_width;
       i += BlockSize) {
    int idx = lod[blockIdx.x] * seq_width + i;
    seq[idx] *= scales[blockIdx.x];
  }
}

template <typename T>
class ScaleLoDTensorFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context, const T* scales,
                  framework::LoDTensor* seq) {
    const size_t level = 0;
    auto lod = seq->lod();
    const size_t num_seq = lod[level].size() - 1;
    const size_t seq_width = seq->numel() / seq->dims()[0];
    auto abs_offset_lod = framework::ToAbsOffset(lod);
    T* seq_data = seq->mutable_data<T>(context.GetPlace());
    paddle::framework::MixVector<size_t> mix_vector(&(abs_offset_lod[level]));

#ifdef PADDLE_WITH_HIP
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(SequenceScaleKernel<T, PADDLE_CUDA_NUM_THREADS>),
        dim3(num_seq), dim3(PADDLE_CUDA_NUM_THREADS), 0, context.stream(),
        seq_data, mix_vector.CUDAMutableData(context.GetPlace()), scales,
        seq_width);
#else
    SequenceScaleKernel<T, PADDLE_CUDA_NUM_THREADS><<<
        num_seq, PADDLE_CUDA_NUM_THREADS, 0, context.stream()>>>(
        seq_data, mix_vector.CUDAMutableData(context.GetPlace()), scales,
        seq_width);
#endif
    mix_vector.CopyToCPU();
  }
};

template class ScaleLoDTensorFunctor<platform::CUDADeviceContext, float>;
template class ScaleLoDTensorFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
