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
#include "paddle/fluid/operators/gather_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_version_registry.h"
namespace paddle {
namespace operators {

template <typename T>
class GatherOpXPUKernel : public framework::OpKernel<T> {
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
    if (index->type() == framework::proto::VarType::INT32) {
      r = xpu::gather<T, int>(dev_ctx.x_context(), x->data<T>(),
                              index->data<int>(), output->data<T>(), xshape,
                              index->dims()[0], 0);
    } else {
      r = xpu::gather<T, int64_t>(dev_ctx.x_context(), x->data<T>(),
                                  index->data<int64_t>(), output->data<T>(),
                                  xshape, index->dims()[0], 0);
    }
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "XPU gather kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

template <typename T>
class GatherGradOpXPUKernel : public framework::OpKernel<T> {
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
    if (index->type() == framework::proto::VarType::INT32) {
      r = xpu::gather_grad<T, int>(dev_ctx.x_context(), dout->data<T>(),
                                   index->data<int>(), dx->data<T>(), xshape,
                                   index->dims()[0], 0, overwrite);
    } else {
      r = xpu::gather_grad<T, int64_t>(dev_ctx.x_context(), dout->data<T>(),
                                       index->data<int64_t>(), dx->data<T>(),
                                       xshape, index->dims()[0], 0, overwrite);
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
REGISTER_OP_XPU_KERNEL(gather, ops::GatherOpXPUKernel<float>);
REGISTER_OP_XPU_KERNEL(gather_grad, ops::GatherGradOpXPUKernel<float>);
#endif
