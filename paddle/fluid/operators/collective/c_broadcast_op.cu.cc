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

#include "paddle/fluid/operators/collective/c_broadcast_op.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CBroadcastOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int rid = ctx.Attr<int>("ring_id");
    const auto& place = ctx.GetPlace();
    ctx.device_context().Alloc<T>(out);

    int root = ctx.Attr<int>("root");

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      distributed::ProcessGroup* pg = map->get(rid);
      auto b_opts = distributed::BroadcastOptions();
      b_opts.source_rank = rid;
      b_opts.source_root = root;
      auto task = pg->Broadcast(out, *x, b_opts, false);
      task->Wait();
      return;
    }

    gpuStream_t stream = ctx.cuda_device_context().stream();
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (comm_context_manager.Has(std::to_string(rid))) {
      auto* comm_context = static_cast<phi::distributed::NCCLCommContext*>(
          comm_context_manager.Get(std::to_string(rid)));

      comm_context->Broadcast(out, *x, root, stream);
    } else {
      // NOTE(liyurui): This will be removed after moving this operator to phi.
      int numel = x->numel();
      ncclDataType_t dtype = phi::ToNCCLDataType(x->dtype());
      auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
      if (root == comm->rank()) {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclBcast(
            reinterpret_cast<void*>(const_cast<T*>(x->data<T>())),
            numel,
            dtype,
            root,
            comm->comm(),
            stream));
        VLOG(3) << "rank " << comm->rank() << " invoke Bcast. sent "
                << x->numel();
        if (out != x) {
          framework::TensorCopy(*static_cast<const phi::DenseTensor*>(x),
                                place,
                                *phi::DeviceContextPool::Instance().Get(place),
                                static_cast<phi::DenseTensor*>(out));
        }
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::ncclBcast(
            out->data<T>(), numel, dtype, root, comm->comm(), stream));
        VLOG(3) << "rank " << comm->rank() << " invoke Bcast. received "
                << common::product(out->dims());
      }
    }

    out->set_lod(x->lod());
#else
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(c_broadcast,
                          GPU,
                          ALL_LAYOUT,
                          ops::CBroadcastOpCUDAKernel,
                          int,
                          int64_t,
                          float,
                          double,
#if (NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000) || \
    defined(PADDLE_WITH_HIP)
                          phi::dtype::bfloat16,
#endif
                          phi::dtype::float16) {
}
