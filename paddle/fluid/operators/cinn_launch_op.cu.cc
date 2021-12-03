/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/cinn_launch_op.h"
#include <memory>
#include <vector>
#include "cinn/runtime/cinn_runtime.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device/gpu/gpu_types.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace paddle {
namespace operators {
namespace details {

#ifdef PADDLE_WITH_CUDA
void CUDART_CB ReleaseScope(void* data) {
  auto* temp_scope = static_cast<framework::Scope*>(data);
  delete temp_scope;
}

void CUDART_CB ReleaseBuffers(void* data) {
  auto* buffers =
      static_cast<std::vector<std::unique_ptr<cinn_buffer_t>>*>(data);
  delete buffers;
}

template <>
void ReleaseResource<platform::CUDADeviceContext>(
    const std::vector<void*>& resources, void* stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaLaunchHostFunc(
      static_cast<gpuStream_t>(stream), ReleaseScope, resources[0]));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaLaunchHostFunc(
      static_cast<gpuStream_t>(stream), ReleaseBuffers, resources[1]));
}

template <>
void* GetStream<platform::CUDADeviceContext>(
    const framework::ExecutionContext& ctx) {
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  return dev_ctx.stream();
}
#endif

}  // namespace details
}  // namespace operators
}  // namespace paddle

/* see [Why use single type kernel] */
REGISTER_OP_CUDA_KERNEL(cinn_launch,
                        paddle::operators::CinnLaunchOpKernel<
                            paddle::platform::CUDADeviceContext, float>);
