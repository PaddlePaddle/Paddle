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

#include "paddle/fluid/operators/collective/partial_allgather_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class PartialAllGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int64_t numel = in->numel();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);

    PADDLE_ENFORCE_EQ(
        nranks,
        comm->nranks(),
        platform::errors::InvalidArgument(
            "nranks: %s should equal to %s", nranks, comm->nranks()));
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

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      auto task = pg->AllGather(out, *in, offset, send_numel, /*sync_op*/ true);
      task->Wait();
    } else {
      const T* send_buff = in->data<T>() + offset;
      T* recv_buff = out->data<T>();

      gpuStream_t stream = nullptr;
      if (ctx.Attr<bool>("use_calc_stream")) {
        // should ExecutionContext for calc stream.
        stream = ctx.cuda_device_context().stream();
      } else {
        stream = comm->stream();
      }

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::ncclAllGather(send_buff,
                                           recv_buff,
                                           send_numel,
                                           static_cast<ncclDataType_t>(dtype),
                                           comm->comm(),
                                           stream));
    }
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(partial_allgather,
                          GPU,
                          ALL_LAYOUT,
                          ops::PartialAllGatherOpCUDAKernel,
                          float,
                          double,
#if NCCL_VERSION_CODE >= 21000
                          plat::bfloat16,
#endif
                          int,
                          int64_t,
                          plat::float16) {}
