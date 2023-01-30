/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
<<<<<<< HEAD
#include "paddle/phi/core/distributed/comm_context_manager.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

#if defined(PADDLE_WITH_GLOO)
#include <gloo/broadcast.h>

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
<<<<<<< HEAD
#include "paddle/phi/core/distributed/gloo_comm_context.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#endif

namespace paddle {
namespace operators {

template <typename T>
class CBroadcastOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
<<<<<<< HEAD
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    auto root = ctx.Attr<int>("root");

    int rid = ctx.Attr<int>("ring_id");
    ctx.device_context().Alloc<T>(out);

    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (comm_context_manager.Has(rid)) {
      auto* comm_context = static_cast<phi::distributed::GlooCommContext*>(
          comm_context_manager.Get(rid));
      comm_context->Broadcast(out, *in, root);
    } else {
      // NOTE: This will be removed after moving this operator to phi.
      int64_t send_numel = in->numel();
      T* recv_buff = reinterpret_cast<T*>(out->data());
      auto gloo = paddle::framework::GlooWrapper::GetInstance();
      PADDLE_ENFORCE_EQ(
          gloo->IsInitialized(),
          true,
          platform::errors::PreconditionNotMet(
              "You must initialize the gloo environment first to use it."));
      gloo::BroadcastOptions opts(gloo->GetContext());
      opts.setOutput(recv_buff, send_numel);
      opts.setRoot(root);
      gloo::broadcast(opts);
    }
=======
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto root = ctx.Attr<int>("root");

    auto place = ctx.GetPlace();
    int64_t send_numel = in->numel();
    T* recv_buff = out->mutable_data<T>(in->dims(), place);
    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(
        gloo->IsInitialized(),
        true,
        platform::errors::PreconditionNotMet(
            "You must initialize the gloo environment first to use it."));
    gloo::BroadcastOptions opts(gloo->GetContext());
    opts.setOutput(recv_buff, send_numel);
    opts.setRoot(root);
    gloo::broadcast(opts);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
