/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "dgc/dgc.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/fluid/operators/dgc_wait_comm_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class DGCWaitCommOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(place), true,
        platform::errors::PreconditionNotMet(
            "wait_comm op can run on gpu place only for now, but got %s",
            place.DebugString()));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto x = ctx.Input<framework::Tensor>("X");
    int rid = ctx.Attr<int>("ring_id");
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    PADDLE_ENFORCE_EQ(
    map->has(rid), true,
    platform::errors::InvalidArgument("dgc only nomally work after PaddlePaddle==2.3.1"));
    distributed::ProcessGroup* pg = map->get(rid);
    std::vector<phi::DenseTensor> in_tensor = {*x};
    std::vector<std::unique_ptr<phi::GPUContext>> ctxs = pg->GetDeviceContext(in_tensor);

    auto compute_stream =
        static_cast<phi::GPUContext*>(
            platform::DeviceContextPool::Instance().Get(place))
            ->stream();
    auto comm_stream = ctxs[0]->stream();

    gpuEvent_t event; 

    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventCreate(&event, cudaEventDisableTiming));

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipEventRecord(event, comm_stream));
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamWaitEvent(compute_stream, event, 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(event, comm_stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamWaitEvent(compute_stream, event, 0));
#endif
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    dgc_wait_comm,
    ops::DGCWaitCommOpCUDAKernel<phi::GPUContext, float>);
