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

#include "paddle/fluid/operators/collective/barrier_op.h"
#if defined(PADDLE_WITH_CNCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/mlu/cncl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class BarrierOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_CNCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    cnclDataType_t dtype =
        platform::ToCNCLDataType(framework::TransToProtoVarType(in->dtype()));
    int64_t numel = in->numel();
    const void* sendbuff = in->data();
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
    auto cncl_comm = platform::CNCLCommContext::Instance().Get(rid, place);
    auto* comm = cncl_comm->comm();
    auto comm_stream = cncl_comm->stream();
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MLUDeviceContext>();
    cnclReduceOp_t cncl_red_type = cnclSum;
    dev_ctx.Wait();
    PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(
        sendbuff, recvbuff, numel, dtype, cncl_red_type, comm, comm_stream));
    PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueSync(comm_stream));
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should compile with CNCL."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(barrier, ops::BarrierOpMLUKernel<int>);
