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

#include "paddle/fluid/operators/collective/c_split_op.h"
#if defined(PADDLE_WITH_XPU)
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CSplitOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using XPUType = typename XPUTypeTrait<T>::Type;
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");

    PADDLE_ENFORCE_GE(rank,
                      0,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "greater than or equal to 0.",
                          rank));
    PADDLE_ENFORCE_GE(nranks,
                      2,
                      platform::errors::PreconditionNotMet(
                          "The value of nranks (%d) for c_split must be "
                          "greater than or equal to 2.",
                          nranks));
    PADDLE_ENFORCE_LT(rank,
                      nranks,
                      platform::errors::PreconditionNotMet(
                          "The value of rank (%d) for c_split must be "
                          "less than that of nranks (%d).",
                          rank,
                          nranks));

    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();
    auto dims = x->dims();
    auto dims_size = dims.size();
    // final dim
    int64_t end_size = dims[dims_size - 1];

    // remain dim
    auto remain_ddim = phi::slice_ddim(dims, 0, dims_size - 1);
    int64_t remain_numel = phi::product(remain_ddim);

    dims[dims_size - 1] /= nranks;
    out->Resize(dims);
    dev_ctx.template Alloc(out, x->dtype());

    std::vector<XPUType*> output_list(nranks, nullptr);
    output_list.at(rank) = reinterpret_cast<XPUType*>(out->data<T>());
    std::vector<int64_t> split_list(nranks, dims[dims_size - 1]);
    int axis = 1;

    auto ret = xpu::split(dev_ctx.x_context(),
                          reinterpret_cast<const XPUType*>(x->data<T>()),
                          output_list,
                          {remain_numel, end_size},
                          split_list,
                          axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "split");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(c_split,
                          XPU,
                          ALL_LAYOUT,
                          ops::CSplitOpXPUKernel,
                          float,
                          int,
                          plat::float16) {}
