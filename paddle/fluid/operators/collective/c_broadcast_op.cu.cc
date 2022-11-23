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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
class CBroadcastOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int numel = x->numel();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(*x);
      out_tensor.push_back(*out);
      auto task = pg->Broadcast(in_tensor, out_tensor);
      task->Wait();
      return;
    }

    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    gpuStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::GPUContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    int root = ctx.Attr<int>("root");
    if (root == comm->rank()) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclBcast(
          reinterpret_cast<void*>(const_cast<T*>(x->data<T>())),
          numel,
          dtype,
          root,
          comm->comm(),
          stream));
      VLOG(3) << "rank " << comm->rank() << " invoke Bcast. sent "
              << x->numel();

      if (out != x) {
        framework::TensorCopy(
<<<<<<< HEAD
            *static_cast<const framework::Tensor*>(x),
=======
            *static_cast<const phi::DenseTensor*>(x),
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            place,
            *platform::DeviceContextPool::Instance().Get(place),
            static_cast<phi::DenseTensor*>(out));
      }
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::ncclBcast(out->mutable_data<T>(place),
                                       numel,
                                       dtype,
                                       root,
                                       comm->comm(),
                                       stream));
      VLOG(3) << "rank " << comm->rank() << " invoke Bcast. received "
              << phi::product(out->dims());
    }

    out->Resize(x->dims());
    out->set_lod(x->lod());
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

REGISTER_OP_CUDA_KERNEL(c_broadcast,
                        ops::CBroadcastOpCUDAKernel<float>,
                        ops::CBroadcastOpCUDAKernel<double>,
<<<<<<< HEAD
#if CUDNN_VERSION_MIN(8, 1, 0) && NCCL_VERSION_CODE >= 21000
=======
#if NCCL_VERSION_CODE >= 21000
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                        ops::CBroadcastOpCUDAKernel<plat::bfloat16>,
#endif
                        ops::CBroadcastOpCUDAKernel<int>,
                        ops::CBroadcastOpCUDAKernel<int64_t>,
                        ops::CBroadcastOpCUDAKernel<plat::float16>);
