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

#include "paddle/fluid/operators/collective/c_allgather_op.h"

#if defined(PADDLE_WITH_CNCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/mlu/cncl_helper.h"
#endif
#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace operators {

template <typename T>
class CAllGatherOpMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_CNCL)
    auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    cnclDataType_t dtype =
        platform::ToCNCLDataType(framework::TransToProtoVarType(x->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::CNCLCommContext::Instance().Get(rid, place);
    PADDLE_ENFORCE_EQ(
        nranks, comm->nranks(),
        platform::errors::InvalidArgument("nranks: %s should equal to %s",
                                          nranks, comm->nranks()));

    framework::DDim out_dims = x->dims();
    out_dims[0] *= nranks;
    out->mutable_data<T>(out_dims, place);

    uint32_t send_numel = x->numel();
    void* send_buff = reinterpret_cast<void*>(const_cast<T*>(x->data<T>()));
    void* recv_buff = reinterpret_cast<void*>(out->data<T>());

    mluStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::MLUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(send_buff, recv_buff, send_numel,
                                             dtype, comm->comm(), stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with MLU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(c_allgather, ops::CAllGatherOpMLUKernel<float>,
                       ops::CAllGatherOpMLUKernel<uint8_t>,
                       ops::CAllGatherOpMLUKernel<int>,
                       ops::CAllGatherOpMLUKernel<int8_t>,
                       ops::CAllGatherOpMLUKernel<int16_t>,
                       ops::CAllGatherOpMLUKernel<plat::float16>);
