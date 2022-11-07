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

#include <memory>

#include "paddle/fluid/operators/collective/partial_allgather_op.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/hccl_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class CallPartialGatherOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int64_t numel = in->numel();
    HcclDataType dtype =
        platform::ToHCCLDataType(framework::TransToProtoVarType(in->dtype()));

    int rank = ctx.Attr<int>("rank");
    int ring_id = ctx.Attr<int>("ring_id");
    std::string group =
        std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    auto place = ctx.GetPlace();
    auto comm = platform::HCCLCommContext::Instance().Get(ring_id, place);
    int nranks = comm->nranks();

    PADDLE_ENFORCE_EQ(rank,
                      comm->rank(),
                      platform::errors::InvalidArgument(
                          "rank: %s should equal to %s", rank, comm->rank()));
    PADDLE_ENFORCE_EQ(
        (numel % nranks),
        0,
        platform::errors::InvalidArgument(
            "The input numel (%d) must be divisible by nranks(%d)",
            numel,
            nranks));

    framework::DDim dims = in->dims();
    out->mutable_data<T>(dims, place);

    int64_t send_numel = numel / nranks;
    int offset = send_numel * rank;

    void *send_buff =
        reinterpret_cast<void *>(const_cast<T *>(in->data<T>()) + offset);
    void *recv_buff = reinterpret_cast<void *>(out->data<T>());

    aclrtStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::NPUDeviceContext *>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    VLOG(3) << "begin hccl allgather, parameter is: "
            << ", group is " << group << ", ring_id is " << ring_id
            << ", nranks is " << nranks << ", rankid is " << rank;

    PADDLE_ENFORCE_NPU_SUCCESS(
        platform::dynload::HcclAllGather(send_buff,
                                         recv_buff,
                                         send_numel,
                                         dtype,
                                         comm->comm(),
                                         reinterpret_cast<void *>(stream)));

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

REGISTER_OP_NPU_KERNEL(partial_allgather,
                       ops::CallPartialGatherOpASCENDKernel<int8_t>,
                       ops::CallPartialGatherOpASCENDKernel<int>,
                       ops::CallPartialGatherOpASCENDKernel<float>,
                       ops::CallPartialGatherOpASCENDKernel<plat::float16>);
