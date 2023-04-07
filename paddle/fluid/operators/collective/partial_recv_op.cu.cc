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

#include "paddle/fluid/operators/collective/partial_recv_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class PartialRecvOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
    auto out = ctx.Output<phi::DenseTensor>("Out");
    auto out_dims = out->dims();
    auto numel = out->numel();

    int rid = ctx.Attr<int>("ring_id");
    int peer = ctx.Attr<int>("peer");
    int data_type = ctx.Attr<int>("dtype");
    int num = ctx.Attr<int>("num");
    int id = ctx.Attr<int>("id");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(data_type);

    PADDLE_ENFORCE_GE(
        rid,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for partial_recv op must be non-negative.", rid));
    PADDLE_ENFORCE_GE(
        peer,
        0,
        platform::errors::InvalidArgument(
            "The peer (%d) for partial_recv op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(num,
                      1,
                      platform::errors::InvalidArgument(
                          "The num (%d) for partial_recv op must >=1", num));
    PADDLE_ENFORCE_EQ(
        (id >= 0 && id < num),
        true,
        platform::errors::InvalidArgument(
            "The id (%d) for partial_recv op must >=0 and <num (%d)", id, num));
    PADDLE_ENFORCE_EQ(
        (numel % num),
        0,
        platform::errors::InvalidArgument(
            "The input numel (%d) must be divisible by num(%d)", numel, num));

    auto place = ctx.GetPlace();
    out->mutable_data<T>(out_dims, place);
    int64_t recv_numel = numel / num;
    int64_t offset = recv_numel * id;

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup *pg = map->get(rid);
      auto task = pg->Recv(out, peer, offset, recv_numel, /*sync_op*/ true);
      task->Wait();
    } else {
      gpuStream_t stream = nullptr;
      auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
      if (ctx.Attr<bool>("use_calc_stream")) {
        // should ExecutionContext for calc stream.
        stream = ctx.cuda_device_context().stream();
      } else {
        stream = comm->stream();
      }
      PADDLE_ENFORCE_LT(peer,
                        comm->nranks(),
                        platform::errors::InvalidArgument(
                            "The value of peer (%d) you set must "
                            "be less than comm->nranks (%d).",
                            peer,
                            comm->nranks()));
      ncclDataType_t dtype = platform::ToNCCLDataType(type);
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::ncclRecv(out->data<T>() + offset,
                                      recv_numel,
                                      dtype,
                                      peer,
                                      comm->comm(),
                                      stream));
      VLOG(3) << "rank " << comm->rank() << " recv " << recv_numel
              << " from offset[" << offset << "] from " << peer;
    }
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should be compiled with NCCL and "
        "NCCL version >= 2.7.3 is needed."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(partial_recv,
                          GPU,
                          ALL_LAYOUT,
                          ops::PartialRecvOpCUDAKernel,
                          float,
                          double,
#if NCCL_VERSION_CODE >= 21000
                          plat::bfloat16,
#endif
                          int,
                          int64_t,
                          plat::float16) {}
