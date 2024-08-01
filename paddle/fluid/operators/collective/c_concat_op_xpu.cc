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
#include "paddle/phi/core/distributed/comm_context_manager.h"

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/kernels/funcs//concat_and_split_functor.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/common/flags.h"
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CConcatOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    BKCLDataType dtype =
        platform::ToBKCLDataType(framework::TransToProtoVarType(x->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(rank,
                      0,
                      common::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "greater than or equal to 0.",
                          rank));
    PADDLE_ENFORCE_GE(nranks,
                      2,
                      common::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_concat must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank,
                      nranks,
                      common::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "less than that of nranks (%d).",
                          rank,
                          nranks));

#if defined(PADDLE_WITH_XPU_BKCL)
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();
    phi::DenseTensor temp_out;
    phi::DDim temp_out_dims = x->dims();
    temp_out_dims[0] *= nranks;
    temp_out.Resize(temp_out_dims);
    dev_ctx.template Alloc(&temp_out, x->dtype());

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    int64_t send_numel = x->numel();
    const T* send_buff = x->data<T>();
    T* recv_buff = temp_out.data<T>();
    XPUStream stream = nullptr;

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
      platform::BKCLComm* comm = nullptr;
      phi::distributed::BKCLCommContext* comm_ctx = nullptr;
      const auto& comm_context_manager =
          phi::distributed::CommContextManager::GetInstance();
      if (FLAGS_dynamic_static_unified_comm) {
        PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(rid)),
                          true,
                          common::errors::InvalidArgument(
                              "You choose to use new communication library by "
                              "setting environment "
                              "variable FLAGS_dynamic_static_unified_comm "
                              "True. But ring_id(%d) is "
                              "not found in comm_context_manager.",
                              std::to_string(rid)));
        comm_ctx = static_cast<phi::distributed::BKCLCommContext*>(
            comm_context_manager.Get(std::to_string(rid)));
        PADDLE_ENFORCE_NE(
            comm_ctx,
            nullptr,
            common::errors::Unavailable(
                "BKCLCommContext is nullptr, collective op should "
                "has ring_id attr."));
        stream = comm_ctx->GetStream();
        VLOG(3) << "new comm_context_manager has rid " << rid;
      } else {  // old comm_context
        auto place = ctx.GetPlace();
        comm = platform::BKCLCommContext::Instance().Get(rid, place);
        PADDLE_ENFORCE_EQ(
            nranks,
            comm->nranks(),
            common::errors::InvalidArgument(
                "nranks: %s should equal to %s", nranks, comm->nranks()));
        stream = comm->stream();
        VLOG(3) << "old BKCLCommContext has rid " << rid;
      }
      if (ctx.Attr<bool>("use_calc_stream")) {
        stream = dev_ctx.x_context()->xpu_stream;
      }
      if (comm_ctx) {
        comm_ctx->AllGather(&temp_out, *x, stream);
      } else {
        PADDLE_ENFORCE_XPU_SUCCESS(bkcl_all_gather(
            comm->comm(), send_buff, send_numel, recv_buff, dtype, stream));
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

    phi::funcs::ConcatFunctor<phi::XPUContext, T> functor;
    out->Resize(out_dims);
    dev_ctx.template Alloc(out, x->dtype());
    functor(dev_ctx, inputs, axis, out);
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with XPU."));
#endif
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(c_concat,
                          XPU,
                          ALL_LAYOUT,
                          ops::CConcatOpXPUKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
