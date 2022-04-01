/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/ddim.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class GatherOpXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on XPU."));

    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *output = ctx.Output<Tensor>("Out");
    if (ctx.HasInput("Axis")) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Now, it doesn't support XPU with Axis."));
    }

    output->mutable_data<T>(ctx.GetPlace());
    if (x->numel() == 0) return;

    const auto index_dims = index->dims();
    if (index_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          index_dims[1], 1,
          platform::errors::InvalidArgument(
              "The last dim of index should be 1 when it is 2D, but we get %d",
              index_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          index_dims.size(), 1,
          platform::errors::InvalidArgument(
              "The index should be 1D, when it is not 2D, but we get %d",
              index_dims.size()));
    }
    std::vector<int> xshape(x->dims().size());
    for (int i = 0; i < x->dims().size(); ++i) {
      xshape[i] = x->dims()[i];
    }

    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    int r = XPU_SUCCESS;
    if (framework::TransToProtoVarType(index->dtype()) ==
        framework::proto::VarType::INT32) {
      r = xpu::gather<XPUType, int>(
          dev_ctx.x_context(), reinterpret_cast<const XPUType *>(x->data<T>()),
          index->data<int>(), reinterpret_cast<XPUType *>(output->data<T>()),
          xshape, index->dims()[0], 0);
    } else {
      r = xpu::gather<XPUType, int64_t>(
          dev_ctx.x_context(), reinterpret_cast<const XPUType *>(x->data<T>()),
          index->data<int64_t>(),
          reinterpret_cast<XPUType *>(output->data<T>()), xshape,
          index->dims()[0], 0);
    }
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "XPU gather kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

template <typename T>
class GatherGradOpXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on XPU."));

    auto *index = ctx.Input<Tensor>("Index");
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

    if (ctx.HasInput("Axis")) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Now, it doesn't support XPU with Axis."));
    }
    if (dout->numel() == 0) {
      return;
    }

    bool overwrite = ctx.Attr<bool>("overwrite");
    const auto index_dims = index->dims();
    if (index_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          index_dims[1], 1,
          platform::errors::InvalidArgument(
              "The last dim of index should be 1 when it is 2D, but we get %d",
              index_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          index_dims.size(), 1,
          platform::errors::InvalidArgument(
              "The index should be 1D, when it is not 2D, but we get %d",
              index_dims.size()));
    }
    std::vector<int> xshape(dx->dims().size());
    for (int i = 0; i < dx->dims().size(); ++i) {
      xshape[i] = dx->dims()[i];
    }

    dx->mutable_data<T>(ctx.GetPlace());

    int r = XPU_SUCCESS;
    if (framework::TransToProtoVarType(index->dtype()) ==
        framework::proto::VarType::INT32) {
      r = xpu::gather_grad<XPUType, int>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType *>(dout->data<T>()),
          index->data<int>(), reinterpret_cast<XPUType *>(dx->data<T>()),
          xshape, index->dims()[0], 0, overwrite);
    } else {
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      int *index_int_ptr_l3 =
          RAII_GUARD.alloc_l3_or_gm<int32_t>(index->numel());
      r = xpu::cast_v2<int64_t, int32_t>(dev_ctx.x_context(),
                                         index->data<int64_t>(),
                                         index_int_ptr_l3, index->numel());
      PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                            "XPU API(cast_v2) return wrong "
                                            "value[%d %s]",
                                            r, XPUAPIErrorMsg[r]));

      r = xpu::gather_grad<XPUType, int>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType *>(dout->data<T>()), index_int_ptr_l3,
          reinterpret_cast<XPUType *>(dx->data<T>()), xshape, index->dims()[0],
          0, overwrite);
    }
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "XPU gather grad kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(gather, ops::GatherOpXPUKernel<float>,
                       ops::GatherOpXPUKernel<paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(gather_grad, ops::GatherGradOpXPUKernel<float>,
                       ops::GatherGradOpXPUKernel<paddle::platform::float16>);
#endif
