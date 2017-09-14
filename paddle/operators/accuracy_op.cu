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

#include "paddle/operators/accuracy_op.h"

namespace paddle {
namespace operators {

__global__ void AccuracyCudaKernel(const int N, const int D, const int* Xdata,
                                   const int* labeldata, float* accuracy) {
  int count = 0;
  __shared__ int total;
  total = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (N);
       i += blockDim.x * gridDim.x) {
    for (int j = 0; j < D; ++j) {
      if (Xdata[i * D + j] == labeldata[i]) {
        ++count;
        break;
      }
    }
  }
  atomicAdd(&total, count);
  __syncthreads();
  if (threadIdx.x == 0) {
    *accuracy = static_cast<float>(total) / static_cast<float>(N);
  }
}

template <typename T>
class AccuracyOpCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");
    auto* inference = ctx.Input<Tensor>("Inference");
    auto* label = ctx.Input<Tensor>("Label");
    auto* accuracy = ctx.Output<Tensor>("Accuracy");
    // FIXME(typhoonzero): only support indices currently
    // if add support for output values, how to detect the data type?
    const int* inference_data = inference->data<int>();
    const int* label_data = label->data<int>();
    float* accuracy_data = accuracy->mutable_data<float>(ctx.GetPlace());

    size_t num_samples = inference->dims()[0];
    size_t infer_width = inference->dims()[1];
    cudaMemset((void**)&accuracy_data, 0, sizeof(float));

    if (num_samples == 0) {
      return;
    }

    int threads = 512;
    int grids = (num_samples + 4096 - 1) / 4096;
    AccuracyCudaKernel<<<grids, threads>>>(
        num_samples, infer_width, inference_data, label_data, accuracy_data);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_GPU_KERNEL(accuracy,
                       paddle::operators::AccuracyOpCUDAKernel<float>);
