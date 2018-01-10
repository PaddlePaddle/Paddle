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

#include "paddle/operators/math/sequence_scale.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__global__ void SequenceScaleKernel(T* seq, size_t* lod, const T* scales,
                                    const size_t num_seq,
                                    const size_t seq_width) {
  size_t idx = blockIdx.x * blockDim.y + threadIdx.y;

  if (idx < lod[num_seq]) {
    size_t i = 0;
    for (i = 0; i < num_seq; ++i) {
      if (idx < lod[i + 1] * seq_width) {
        break;
      }
    }
    seq[i] *= scales[i];
  }
}

template <typename T>
class ScaleLoDTensorFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  framework::LoDTensor& seq, const T* scales,
                  const size_t num_seq) {
    auto lod = seq.lod();
    const size_t seq_width = seq.dims()[1];
    const size_t level = 0;
    framework::LoD abs_offset_lod = framework::ToAbsOffset(lod);
    T* seq_data = seq.mutable_data<T>(context.GetPlace());

    int threads = 1024;
    int grid = (seq.numel() * seq_width + threads - 1) / threads;
    SequenceScaleKernel<T><<<grid, threads, 0, context.stream()>>>(
        seq_data, abs_offset_lod[level].data(), scales, num_seq, seq_width);
  }
};

template class ScaleLoDTensorFunctor<platform::CUDADeviceContext, float>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
