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

#include "paddle/fluid/operators/collective/c_allgather_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/distributed/collective/ProcessGroupStream.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
class CAllGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    int rid = ctx.Attr<int>("ring_id");
    const auto& map = distributed::ProcessGroupIdMap::GetInstance();
    if (map.find(rid) != map.end()) {
      const auto& group =
          std::static_pointer_cast<distributed::ProcessGroupStream>(
              map.at(rid));

      const auto& in = ctx.Input<phi::DenseTensor>("X");
      auto* out = ctx.Output<phi::DenseTensor>("Out");
      bool use_calc_stream = ctx.Attr<bool>("use_calc_stream");
      bool sync_op = use_calc_stream;

      // Allocate memory for out
      const auto& place = ctx.GetPlace();
      int nranks = ctx.Attr<int>("nranks");
      framework::DDim out_dims = in->dims();
      out_dims[0] *= nranks;
      out->mutable_data<T>(out_dims, place);

      group->AllGather(out, *in, 0, -1, sync_op, use_calc_stream);
      return;
    }

    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    PADDLE_ENFORCE_EQ(
        nranks,
        comm->nranks(),
        platform::errors::InvalidArgument(
            "nranks: %s should equal to %s", nranks, comm->nranks()));

    framework::DDim out_dims = in->dims();
    out_dims[0] *= nranks;
    out->mutable_data<T>(out_dims, place);

    int64_t send_numel = in->numel();
    const T* send_buff = in->data<T>();
    T* recv_buff = out->data<T>();

    gpuStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::GPUContext*>(dev_ctx)->stream();
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

REGISTER_OP_CUDA_KERNEL(c_allgather,
                        ops::CAllGatherOpCUDAKernel<float>,
                        ops::CAllGatherOpCUDAKernel<double>,
#if NCCL_VERSION_CODE >= 21000
                        ops::CAllGatherOpCUDAKernel<plat::bfloat16>,
#endif
                        ops::CAllGatherOpCUDAKernel<int>,
                        ops::CAllGatherOpCUDAKernel<uint8_t>,
                        ops::CAllGatherOpCUDAKernel<int8_t>,
                        ops::CAllGatherOpCUDAKernel<int64_t>,
                        ops::CAllGatherOpCUDAKernel<bool>,
                        ops::CAllGatherOpCUDAKernel<plat::float16>);
