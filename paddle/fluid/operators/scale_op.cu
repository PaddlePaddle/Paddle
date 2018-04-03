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

#include <algorithm>
#include <list>
#include "paddle/fluid/operators/scale_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void scale_kernel(const T* x, const size_t numel, const float scale,
                             T* out) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < numel;
       idx += blockDim.x * gridDim.x) {
    out[idx] = scale * x[idx];
  }
}

template <typename T>
class ScaleKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    std::list<uint64_t> local_times;
    local_times.push_back(PosixInNsec());
    auto* tensor = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    auto scale = static_cast<T>(context.Attr<float>("scale"));
    tensor->mutable_data<T>(in->place());
    local_times.push_back(PosixInNsec());

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    int64_t max_threads =
        static_cast<int64_t>(dev_ctx.GetMaxPhysicalThreadCount());
    int block =
        std::max(std::min((in->numel() + 1) / 1024, max_threads / 1024), 1L);

    scale_kernel<T><<<block, 1024, 0, dev_ctx.stream()>>>(
        in->data<T>(), in->numel(), scale, tensor->data<T>());
    local_times.push_back(PosixInNsec());
    Times.push_back(local_times);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    scale,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, float>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, double>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, int>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>);
