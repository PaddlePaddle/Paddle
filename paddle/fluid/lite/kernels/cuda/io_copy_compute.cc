// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/cuda/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using TargetW = TargetWrapper<TARGET(kHost), cudaStream_t, cudaEvent_t>;

// Host to CUDA memory.
void CopyFromHostSync(void* target, const void* source, size_t size) {
  TargetW::MemcpySync(target, source, size, IoDirection::HtoD);
}

void CopyFromHostAsync(void* target, const void* source, size_t size,
                       TargetW::stream_t stream) {
  TargetW::MemcpyAsync(target, source, size, IoDirection::HtoD, stream);
}

// Host to Host memory.
void CopyToHostSync(void* target, const void* source, size_t size) {
  TargetW::MemcpySync(target, source, size, IoDirection::DtoH);
}

/*
 * This kernel copies a tensor from host to CUDA space.
 */
class IoCopyHostToCudaCompute
    : public OpKernel<TARGET(kCUDA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kX86));
    auto* data = param.y->mutable_data(target(), param.x->memory_size());
    CopyFromHostSync(data, param.x->data<void>(), param.x->memory_size());
  }
};

/*
 * This kernel copies a tensor from CUDA to host space.
 */
class IoCopyCudaToHostCompute
    : public OpKernel<TARGET(kCUDA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kCUDA));
    auto* data = param.y->mutable_data(TARGET(kHost), param.x->memory_size());
    CopyToHostSync(data, param.x->data<void>(), param.x->memory_size());
  }
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(io_copy, kCUDA, kAny,
                     paddle::lite::kernels::cuda::IoCopyHostToCudaCompute,
                     host_to_device)
    .BindInput("Input", {paddle::lite::Type::Get<paddle::lite::TensorAnyTy>(
                            TARGET(kHost))})
    .BindOutput("Out", {paddle::lite::Type::Get<paddle::lite::TensorAnyTy>(
                           TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy, kCUDA, kAny,
                     paddle::lite::kernels::cuda::IoCopyCudaToHostCompute,
                     device_to_host)
    .BindInput("Input", {paddle::lite::Type::Get<paddle::lite::TensorAnyTy>(
                            TARGET(kCUDA))})
    .BindOutput("Out", {paddle::lite::Type::Get<paddle::lite::TensorAnyTy>(
                           TARGET(kHost))})
    .Finalize();
