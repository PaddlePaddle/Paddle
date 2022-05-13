/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/fluid/framework/convert_utils.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
class NCCLBroadcastOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet(
            "The place of ExecutionContext should be CUDAPlace."));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    int dev_id = ctx.GetPlace().device;
    int root_dev_id = ctx.Attr<int>("root");

    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    PADDLE_ENFORCE_EQ(
        out->IsInitialized(), true,
        platform::errors::PreconditionNotMet(
            "Currently, the output of broadcast op must be initialized,"
            "because this op can only be an In-Place operation."));
    void* send_recv_buffer = out->mutable_data<T>(ctx.GetPlace());
    PADDLE_ENFORCE_EQ(
        send_recv_buffer, in->data(),
        platform::errors::PreconditionNotMet("Currently, the broadcast op can "
                                             "only be an In-Place operation."));

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto comm = dev_ctx.nccl_comm();
    auto stream = dev_ctx.stream();

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclBcast(
        send_recv_buffer, static_cast<size_t>(in->numel()),
        platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype())),
        root_dev_id, comm, stream));

    VLOG(3) << "Bcast " << ctx.InputNames("X")[0] << ", (" << in->numel() << ")"
            << " From " << root_dev_id << " to " << dev_id;

    if (ctx.Attr<bool>("sync_mode")) {
      platform::GpuStreamSync(stream);
    }
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(broadcast, ops::NCCLBroadcastOpKernel<float>,
                        ops::NCCLBroadcastOpKernel<double>,
                        ops::NCCLBroadcastOpKernel<int>,
                        ops::NCCLBroadcastOpKernel<int64_t>,
                        ops::NCCLBroadcastOpKernel<plat::float16>);
