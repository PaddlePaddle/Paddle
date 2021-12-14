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

#include <vector>

#include "paddle/fluid/operators/collective/c_concat_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CConcatOpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    HcclDataType dtype = platform::ToHCCLDataType(x->type());
    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE_GE(rank, 0,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "greater than or equal to 0.",
                          rank));
    PADDLE_ENFORCE_GE(nranks, 2,
                      platform::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_concat must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank, nranks,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_concat must be "
                          "less than that of nranks (%d).",
                          rank, nranks));

#if defined(PADDLE_WITH_ASCEND_CL)
    auto comm = platform::HCCLCommContext::Instance().Get(rid, place);
    PADDLE_ENFORCE_EQ(
        nranks, comm->nranks(),
        platform::errors::InvalidArgument("nranks: %s should equal to %s",
                                          nranks, comm->nranks()));

    framework::Tensor temp_out;
    framework::DDim temp_out_dims = x->dims();
    temp_out_dims[0] *= nranks;
    temp_out.mutable_data<T>(temp_out_dims, place);
    int64_t send_numel = x->numel();
    void *send_buff = reinterpret_cast<void *>(const_cast<T *>(x->data<T>()));
    void *recv_buff = reinterpret_cast<void *>(temp_out.data<T>());
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    auto stream = static_cast<platform::NPUDeviceContext *>(dev_ctx)->stream();

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllGather(
        send_buff, recv_buff, send_numel, dtype, comm->comm(),
        reinterpret_cast<void *>(stream)));

    std::vector<framework::Tensor> inputs;
    int axis = x->dims().size() - 1;
    auto out_dims = x->dims();
    out_dims[out_dims.size() - 1] *= nranks;
    int rows_per_tensor = x->dims()[0];
    int offset = 0;
    std::vector<std::string> names;
    for (int i = 0; i < nranks; i++) {
      framework::Tensor temp = temp_out.Slice(offset, offset + rows_per_tensor);
      inputs.emplace_back(temp);
      offset += rows_per_tensor;
      names.push_back("x" + std::to_string(i));
    }
    out->mutable_data<T>(out_dims, place);

    NpuOpRunner runner{"ConcatD",
                       {inputs},
                       {*out},
                       { {"concat_dim", axis},
                         { "N",
                           static_cast<int>(inputs.size()) } }};
    runner.AddInputNames(names);
    runner.Run(stream);

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

REGISTER_OP_NPU_KERNEL(c_concat, ops::CConcatOpNPUKernel<float>,
                       ops::CConcatOpNPUKernel<int>,
                       ops::CConcatOpNPUKernel<int8_t>,
                       ops::CConcatOpNPUKernel<plat::float16>);
