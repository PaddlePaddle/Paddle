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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#endif

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
class BKCLBroadcastOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_xpu_place(ctx.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "The place of ExecutionContext should be XPUPlace."));

#if defined(PADDLE_WITH_XPU_BKCL)
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

    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    auto comm = dev_ctx.bkcl_context();
    auto stream = dev_ctx.x_context()->xpu_stream;

    // TODO(wangxi16): bkcl_broadcast only support float type,
    // need to converted other type to float before broadcasting.
    // Broadcast is equivalent to no type of operation, does not affect
    // correctness.
    // Once bkcl_broadcast support other type, need chang to:
    // BKCLDataType data_type = platform::ToBKCLDataType(in->type());
    BKCLDataType data_type = BKCL_FLOAT;
    size_t scale = sizeof(T) / sizeof(float);
    auto ret = bkcl_broadcast(comm, send_recv_buffer, send_recv_buffer,
                              static_cast<size_t>(in->numel()) * scale,
                              data_type, root_dev_id, stream);
    PADDLE_ENFORCE_EQ(ret, BKCL_SUCCESS,
                      platform::errors::Unavailable("bkcl_broadcast failed"));

    VLOG(3) << "Bcast " << ctx.InputNames("X")[0] << ", (" << in->numel() << ")"
            << " From " << root_dev_id << " to " << dev_id;

    if (ctx.Attr<bool>("sync_mode")) {
      dev_ctx.Wait();
    }
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with XPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(broadcast, ops::BKCLBroadcastOpKernel<float>,
                       ops::BKCLBroadcastOpKernel<double>,
                       ops::BKCLBroadcastOpKernel<int>,
                       ops::BKCLBroadcastOpKernel<int64_t>);
