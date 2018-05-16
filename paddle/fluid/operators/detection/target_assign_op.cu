/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/target_assign_op.h"

namespace paddle {
namespace operators {

template <typename T, typename WT>
__global__ void NegTargetAssignKernel(const int* neg_indices, const size_t* lod,
                                      const int N, const int M, const int K,
                                      const int mismatch_value, T* out,
                                      WT* out_wt) {
  int bidx = blockIdx.x;
  int st = lod[bidx];
  int ed = lod[bidx + 1];

  int row_start = bidx * M;
  for (int i = st + threadIdx.x; i < ed; i += blockDim.x) {
    int id = row_start + neg_indices[i];
    for (int k = 0; k < K; ++k) {
      out[id * K + k] = T(mismatch_value);
      out_wt[id * K + k] = WT(1.);
    }
  }
}

template <typename T, typename WT>
struct NegTargetAssignFunctor<platform::CUDADeviceContext, T, WT> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const int* neg_indices, const size_t* lod, const int N,
                  const int M, const int K, const int mismatch_value, T* out,
                  WT* out_wt) {
    const int block_size = 256;
    const int grid_size = N;
    NegTargetAssignKernel<T, WT><<<grid_size, block_size, 0, ctx.stream()>>>(
        neg_indices, lod, N, M, K, mismatch_value, out, out_wt);
  }
};

template struct NegTargetAssignFunctor<platform::CUDADeviceContext, int, float>;
template struct NegTargetAssignFunctor<platform::CUDADeviceContext, float,
                                       float>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    target_assign,
    ops::TargetAssignKernel<paddle::platform::CUDADeviceContext, int, float>,
    ops::TargetAssignKernel<paddle::platform::CUDADeviceContext, float, float>);
