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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_GLOO)
#include <gloo/allgather.h>
#include <gloo/barrier.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/http_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/device.h>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CAllGatherOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    framework::DDim out_dims = in->dims();
    auto place = ctx.GetPlace();

    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    auto nranks = gloo->Size();
    out_dims[0] *= nranks;
    int64_t send_numel = in->numel();
    VLOG(0) << "In:";
    const T* send_buff = in->data<T>();
    VLOG(0) << "Out:";
    out->mutable_data<T>(place);
    out->Resize(out_dims);
    T* recv_buff = out->mutable_data<T>(place);
    VLOG(0) << "In_Out:";

    PADDLE_ENFORCE_EQ(
        gloo->IsInitialized(), true,
        platform::errors::InvalidArgument(
            "You must initialize the gloo environment first to use it."));
    gloo::AllgatherOptions opts(gloo->GetContext());
    opts.setInput(const_cast<T*>(send_buff), send_numel);
    opts.setOutput(recv_buff, send_numel * nranks);
    gloo::allgather(opts);
#else
    PADDLE_ENFORCE_EQ(
        true, false,
        platform::errors::Unavailable(
            "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
