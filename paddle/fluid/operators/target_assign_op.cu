/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/target_assign_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void NegTargetAssignKernel(const int* neg_indices, const size_t* lod,
                                      const int num, const int num_prior_box,
                                      const int background_label,
                                      int* out_label, T* out_label_wt) {
  int bidx = blockIdx.x;
  int st = lod[bidx];
  int ed = lod[bidx + 1];

  int row_start = bidx * num_prior_box;
  for (int i = st + threadIdx.x; i < ed; i += blockDim.x) {
    int id = row_start + neg_indices[i];
    out_label[id] = background_label;
    out_label_wt[id] = 1.;
  }
}

template <typename T>
struct NegTargetAssignFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const int* neg_indices, const size_t* lod, const int num,
                  const int num_prior_box, const int background_label,
                  int* out_label, T* out_label_wt) {
    const int block_size = 256;
    const int grid_size = num;
    NegTargetAssignKernel<T><<<grid_size, block_size, 0, ctx.stream()>>>(
        neg_indices, lod, num, num_prior_box, background_label, out_label,
        out_label_wt);
  }
};

template struct NegTargetAssignFunctor<platform::CUDADeviceContext, float>;
template struct NegTargetAssignFunctor<platform::CUDADeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    target_assign,
    ops::TargetAssignKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TargetAssignKernel<paddle::platform::CUDADeviceContext, double>);
