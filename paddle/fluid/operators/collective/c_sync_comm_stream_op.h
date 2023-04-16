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
#ifndef PADDLE_FLUID_OPERATORS_COLLECTIVE_C_SYNC_COMM_STREAM_OP_H_
#define PADDLE_FLUID_OPERATORS_COLLECTIVE_C_SYNC_COMM_STREAM_OP_H_

#include <string>

#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/collective_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CSyncCommStreamKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto place = ctx.GetPlace();
    int ring_id = ctx.Attr<int>("ring_id");
    auto stream =
        platform::NCCLCommContext::Instance().Get(ring_id, place)->stream();

    platform::GpuStreamSync(stream);

#elif defined(PADDLE_WITH_XPU_BKCL)
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_EQ(platform::is_xpu_place(place),
                      true,
                      platform::errors::PreconditionNotMet(
                          "Sync stream op can run on xpu place only for now."));
    int ring_id = ctx.Attr<int>("ring_id");
    auto comm_dev_ctx = platform::BKCLCommContext::Instance()
                            .Get(ring_id, place)
                            ->dev_context();
    comm_dev_ctx->Wait();
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
#endif  // PADDLE_FLUID_OPERATORS_COLLECTIVE_C_SYNC_COMM_STREAM_OP_H_
