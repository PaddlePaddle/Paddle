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

#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/phi/api/include/tensor.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
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
      auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
      PADDLE_ENFORCE_EQ(
          nranks,
          comm->nranks(),
          platform::errors::InvalidArgument(
              "nranks: %s should equal to %s", nranks, comm->nranks()));

      int64_t send_numel = x->numel();
      const T* send_buff = x->data<T>();
      T* recv_buff = temp_out.data<T>();
      gpuStream_t stream = nullptr;
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<phi::GPUContext*>(dev_ctx)->stream();

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::ncclAllGather(send_buff,
                                           recv_buff,
                                           send_numel,
                                           static_cast<ncclDataType_t>(dtype),
                                           comm->comm(),
                                           stream));
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

REGISTER_OP_CUDA_KERNEL(c_concat,
                        ops::CConcatOpCUDAKernel<float>,
                        ops::CConcatOpCUDAKernel<double>,
                        ops::CConcatOpCUDAKernel<int>,
                        ops::CConcatOpCUDAKernel<int64_t>,
                        ops::CConcatOpCUDAKernel<plat::float16>);
