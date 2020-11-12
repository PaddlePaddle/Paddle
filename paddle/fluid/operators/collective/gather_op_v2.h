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

#if defined(PADDLE_WITH_GLOO)
#include <gloo/gather.h>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class GatherOpV2CPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto root_id = ctx.Attr<int>("root");
    auto nranks = ctx.Attr<int>("nranks");

    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(
        gloo->IsInitialized(), true,
        platform::errors::PreconditionNotMet(
            "You must initialize the gloo environment first to use it."));

    PADDLE_ENFORCE_EQ(nranks, gloo->Size(),
                      platform::errors::InvalidArgument(
                          "The number of ranks (%d) you set must "
                          "be equal to gloo->Size() (%d).",
                          nranks, gloo->Size()));
    PADDLE_ENFORCE_GE(
        root_id, 0,
        platform::errors::InvalidArgument(
            "The root_id (%d) for gather_op_v2 must be non-negative.",
            root_id));
    PADDLE_ENFORCE_LT(
        root_id, nranks,
        platform::errors::InvalidArgument(
            "The root_id (%d) for gather_op_v2 must be less than nranks (%d).",
            root_id, nranks));
    int64_t send_numel = in->numel();
    int64_t recv_numel = out->numel();
    auto in_dim = x->dims();
    auto out_dim = framework::DDim(in_dim);
    out_dim[0] *= nranks;
    auto rank = gloo->Rank();
    gloo::GatherOptions opts(gloo->GetContext());
    if (root_id == rank) {
      T* recv_buff = out->mutable_data<T>(place, out_dim);
      opts.setOutput(recv_buff, recv_numel);
    }
    opts.setInput(send_buff, send_numel);
    opts.setRoot(root_id);

    gloo::gather(opts);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
