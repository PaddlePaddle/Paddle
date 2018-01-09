/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/assign_value_op.h"

namespace paddle {
namespace operators {

template <typename T>
class AssignValueGPUKernel : public AssignValueKernel<T> {
 protected:
  virtual void Copy(void* dst, const void* src, size_t size,
                    const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    paddle::platform::GpuMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                     dev_ctx.stream());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(assign_value, ops::AssignValueGPUKernel<int>,
                        ops::AssignValueGPUKernel<float>);
