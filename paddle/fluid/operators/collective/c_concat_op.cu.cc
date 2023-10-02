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

#include "paddle/fluid/operators/collective/c_concat_op.h"

#include <vector>
#include "paddle/fluid/distributed/collective/utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"

#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/phi/api/include/tensor.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/flags.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CConcatOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_GE(rank,
                      0,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "greater than or equal to 0.",
                          rank));
    PADDLE_ENFORCE_GE(nranks,
                      2,
                      platform::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_concat must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank,
                      nranks,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "less than that of nranks (%d).",
                          rank,
                          nranks));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    phi::DenseTensor temp_out;
    framework::DDim temp_out_dims = x->dims();
    temp_out_dims[0] *= nranks;
    temp_out.mutable_data<T>(temp_out_dims, place);

    auto map = distributed::ProcessGroupMapFromGid::getInstance();

    int64_t send_numel = x->numel();
    const T* send_buff = x->data<T>();
    T* recv_buff = temp_out.data<T>();
    gpuStream_t stream = nullptr;

    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(*x);
      out_tensor.push_back(temp_out);
      auto task = pg->AllGather(in_tensor, out_tensor);
      task->Wait();
    } else {
      platform::NCCLComm* comm = nullptr;
      phi::distributed::NCCLCommContext* comm_ctx = nullptr;
      const auto& comm_context_manager =
          phi::distributed::CommContextManager::GetInstance();
      if (FLAGS_dynamic_static_unified_comm) {
        PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(rid)),
                          true,
                          platform::errors::InvalidArgument(
                              "You choose to use new communication library by "
                              "setting environment "
                              "variable FLAGS_dynamic_static_unified_comm "
                              "True. But ring_id(%d) is "
                              "not found in comm_context_manager.",
                              std::to_string(rid)));
        comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
            comm_context_manager.Get(std::to_string(rid)));
        PADDLE_ENFORCE_NE(
            comm_ctx,
            nullptr,
            platform::errors::Unavailable(
                "NCCLCommContext is nullptr, collective op should "
                "has ring_id attr."));
        stream = comm_ctx->GetStream();
        VLOG(3) << "new comm_context_manager has rid " << rid;
      } else {  // old comm_context
        comm = platform::NCCLCommContext::Instance().Get(rid, place);
        PADDLE_ENFORCE_EQ(
            nranks,
            comm->nranks(),
            platform::errors::InvalidArgument(
                "nranks: %s should equal to %s", nranks, comm->nranks()));
        stream = comm->stream();
        VLOG(3) << "old NCCLCommContext has rid " << rid;
      }
      if (ctx.Attr<bool>("use_calc_stream")) {
        // should ExecutionContext for calc stream.
        stream = ctx.cuda_device_context().stream();
      }
      if (comm_ctx) {
        comm_ctx->AllGather(&temp_out, *x, stream);
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::ncclAllGather(send_buff,
                                             recv_buff,
                                             send_numel,
                                             static_cast<ncclDataType_t>(dtype),
                                             comm->comm(),
                                             stream));
      }
    }

    std::vector<phi::DenseTensor> inputs;
    int axis = x->dims().size() - 1;
    auto out_dims = x->dims();
    out_dims[out_dims.size() - 1] *= nranks;
    int rows_per_tensor = x->dims()[0];
    int offset = 0;
    for (int i = 0; i < nranks; i++) {
      phi::DenseTensor temp = temp_out.Slice(offset, offset + rows_per_tensor);
      inputs.emplace_back(temp);
      offset += rows_per_tensor;
    }

    math::ConcatFunctor<phi::GPUContext, T> functor;
    out->mutable_data<T>(out_dims, place);
    auto& dev_ctx2 = ctx.template device_context<phi::GPUContext>();
    functor(dev_ctx2, inputs, axis, out);
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

PD_REGISTER_STRUCT_KERNEL(c_concat,
                          GPU,
                          ALL_LAYOUT,
                          ops::CConcatOpCUDAKernel,
                          float,
                          double,
                          int,
                          int64_t,
#if NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000
                          plat::bfloat16,
#endif
                          plat::float16) {
}
