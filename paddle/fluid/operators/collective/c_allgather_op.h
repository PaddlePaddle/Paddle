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

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include "paddle/phi/core/distributed/gloo_comm_context.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CAllGatherOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
    auto& dev_ctx = ctx.device_context<phi::CPUContext>();
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    out->Resize(in->dims());
    dev_ctx.Alloc<T>(out);
    auto comm_ctx = static_cast<phi::distributed::GlooCommContext*>(
        dev_ctx.GetCommContext());
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      ::common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
    comm_ctx->AllGather(out, *in);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
