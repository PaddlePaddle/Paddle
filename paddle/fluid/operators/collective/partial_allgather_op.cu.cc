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
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class PartialAllGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int rid = ctx.Attr<int>("ring_id");

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      CPartialAllgatherProcessGroupFunctor<phi::GPUContext, T> functor_;
      functor_(ctx);
    } else {
      CPartialAllgatherFunctor<phi::GPUContext, T> functor_;
      functor_(ctx);
    }
  }
};

template <typename T>
struct CPartialAllgatherFunctor<phi::GPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    int64_t numel = in->numel();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);

    PADDLE_ENFORCE_EQ(
        nranks, comm->nranks(),
        platform::errors::InvalidArgument("nranks: %s should equal to %s",
                                          nranks, comm->nranks()));
    PADDLE_ENFORCE_EQ(rank, comm->rank(),
                      platform::errors::InvalidArgument(
                          "rank: %s should equal to %s", rank, comm->rank()));
    PADDLE_ENFORCE_EQ(
        (numel % nranks), 0,
        platform::errors::InvalidArgument(
            "The input numel (%d) must be divisible by nranks(%d)", numel,
            nranks));

    framework::DDim dims = in->dims();
    out->mutable_data<T>(dims, place);

    int64_t send_numel = numel / nranks;
    int offset = send_numel * rank;
    const T* send_buff = in->data<T>() + offset;
    T* recv_buff = out->data<T>();

    gpuStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        send_buff, recv_buff, send_numel, static_cast<ncclDataType_t>(dtype),
        comm->comm(), stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

template <typename T>
struct CPartialAllgatherProcessGroupFunctor<phi::GPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    int64_t numel = in->numel();
    // ncclDataType_t dtype =
    //     platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    // auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    distributed::ProcessGroup* pg = map->get(rid);

    PADDLE_ENFORCE_EQ(
        nranks, pg->GetSize(),
        platform::errors::InvalidArgument("nranks: %s should equal to %s",
                                          nranks, pg->GetRank()));
    PADDLE_ENFORCE_EQ(rank, pg->GetRank(),
                      platform::errors::InvalidArgument(
                          "rank: %s should equal to %s", rank, pg->GetRank()));
    PADDLE_ENFORCE_EQ(
        (numel % nranks), 0,
        platform::errors::InvalidArgument(
            "The input numel (%d) must be divisible by nranks(%d)", numel,
            nranks));

    framework::DDim dims = in->dims();
    out->mutable_data<T>(dims, place);

    int64_t send_numel = numel / nranks;
    int offset = send_numel * rank;

    phi::DenseTensor shared_x;
    shared_x.ShareDataWith(*in);
    shared_x.Resize({numel});

    phi::DenseTensor slice_tensor = shared_x.Slice(offset, offset + send_numel);
    std::vector<phi::DenseTensor> send_in;
    send_in.push_back(slice_tensor);

    std::vector<phi::DenseTensor> v_out;
    v_out.push_back(*out);

    pg->AllGather(send_in, v_out)->Synchronize();

// const T* send_buff = in->data<T>() + offset;
// T* recv_buff = out->data<T>();

// gpuStream_t stream = nullptr;
// if (ctx.Attr<bool>("use_calc_stream")) {
//   auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
//   stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
// } else {
//   stream = comm->stream();
// }

// PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
//     send_buff, recv_buff, send_numel, static_cast<ncclDataType_t>(dtype),
//     comm->comm(), stream));

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

REGISTER_OP_CUDA_KERNEL(partial_allgather,
                        ops::PartialAllGatherOpCUDAKernel<float>,
                        ops::PartialAllGatherOpCUDAKernel<double>,
                        ops::PartialAllGatherOpCUDAKernel<int>,
                        ops::PartialAllGatherOpCUDAKernel<int64_t>,
                        ops::PartialAllGatherOpCUDAKernel<plat::float16>);
