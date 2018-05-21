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

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "paddle/fluid/operators/quad_transform_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {
using platform::PADDLE_CUDA_NUM_THREADS;

template <int T>
__global__ void QuadTransformKernel(const int n, const int h, const int w,
                                    const T* input, T* output) {
  int id_n = threadIdx.x + blockDim.x * blockIdx.x;
  int id_h = threadIdx.y + blockDim.y * blockIdx.y;
  int id_w = threadIdx.z + blockDim.z * blockIdx.z;
  if (idx < n && idy < h && idz < w) {
    int id = id_n * h * w + w * id_h + id_w;
    if (id_n % 2 == 0) {
      output[id] = input[id] + id_w;
    } else {
      output[id] = input[id] + id_h;
    }
  }
}

template <typename T>
class QuadTransfromOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    auto* in = ctx.Input<Tensor>("Input");
    auto in_dims = in->dims();
    const T* in_data = in->data<T>();
    auto* out = ctx.Output<Tensor>("Output");
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    int batch_size = in_dims[0];
    int height = in_dims[2];
    int width = in_dims[3];
    dim3 threadsPerBlock(4, 16, 16);
    dim3 numBlocks((batch_size * 8) / threadsPerBlock.x,
                   height / threadsPerBlock.y, width / threadsPerBlock.z);
    QuadTransfromCudaKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(
        batch_size * 8, height, width, in_data, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(quad_transform, paddle::operators::OpCUDAKernel<float>,
                        paddle::operators::AccuracyOpCUDAKernel<double>);
