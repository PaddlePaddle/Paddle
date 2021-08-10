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

#include "paddle/fluid/operators/collective/c_allgather_op.h"

#include <memory>

#include "paddle/fluid/platform/collective_helper.h"

#if defined(PADDLE_WITH_HCCL)
#include "paddle/fluid/platform/hccl_helper.h"
#endif

#if defined(PADDLE_WITH_ECCL)
#include "paddle/fluid/platform/eccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CAllGatherOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    int ring_id = ctx.Attr<int>("ring_id");
    std::string group =
        std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    auto place = ctx.GetPlace();

#if defined(PADDLE_WITH_HCCL)
    auto dtype = platform::ToHCCLDataType(in->type());
    auto comm =
        paddle::platform::HCCLCommContext::Instance().Get(ring_id, place);
#elif defined(PADDLE_WITH_ECCL)
    auto dtype = platform::ToECCLDataType(in->type());
    auto comm =
        paddle::platform::ECCLCommContext::Instance().Get(ring_id, place);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle collective should compile with eccl or hccl."));
#endif
    int nranks = comm->nranks();

    framework::DDim out_dims = in->dims();
    out_dims[0] *= nranks;
    out->mutable_data<T>(out_dims, place);

    uint64_t send_numel = in->numel();
    void *send_buff = reinterpret_cast<void *>(const_cast<T *>(in->data<T>()));
    void *recv_buff = reinterpret_cast<void *>(out->data<T>());

    aclrtStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::NPUDeviceContext *>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    VLOG(3) << "begin ascend allgather, parameter is: "
            << ", group is " << group << ", ring_id is " << ring_id
            << ", nranks is " << nranks;

#if defined(PADDLE_WITH_HCCL)
    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllGather(
        send_buff, recv_buff, send_numel, dtype, comm->comm(),
        reinterpret_cast<void *>(stream)));
#elif defined(PADDLE_WITH_ECCL)
    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::h_eccl_all_gather(
        send_buff, recv_buff, send_numel, dtype, comm->comm().c_str(),
        reinterpret_cast<void *>(stream)));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet
        "PaddlePaddle collective should compile with eccl or hccl."));
#endif

#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with NPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(c_allgather, ops::CAllGatherOpASCENDKernel<int8_t>,
                       ops::CAllGatherOpASCENDKernel<int>,
                       ops::CAllGatherOpASCENDKernel<float>,
                       ops::CAllGatherOpASCENDKernel<plat::float16>);
