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

#include <memory>
#include <string>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/elementwise/elementwise_npu.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class ElementwiseMinNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    int axis = ctx.Attr<int>("axis");
    bool direct_compute = false;
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    if (x_dims.size() >= y_dims.size()) {
      direct_compute = y_dims == phi::slice_ddim(x_dims, axis, x_dims.size());
    } else {
      direct_compute = x_dims == phi::slice_ddim(y_dims, axis, y_dims.size());
    }
    Tensor transformed_x, transformed_y;
    if (direct_compute) {
      transformed_x.ShareDataWith(*x);
      transformed_y.ShareDataWith(*y);
    } else {
      NpuElementWiseOpBroadcast<T>(
          dev_ctx, x, y, axis, &transformed_x, &transformed_y);
    }
    const auto& runner =
        NpuOpRunner("Minimum", {transformed_x, transformed_y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMinGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    axis = (axis == -1 ? std::abs(x->dims().size() - y->dims().size()) : axis);
    auto stream = dev_ctx.stream();
    if (dx && dy) {
      // dx
      dx->mutable_data<T>(ctx.GetPlace());
      Tensor tmp_x;
      tmp_x.ShareDataWith(*dx);
      if (dx->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec_x;
        std::vector<int> reduce_axes_x;
        auto src_dims_x = dx->dims();
        auto dout_dims = dout->dims();

        int src_axis_x = (src_dims_x.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis_x || ax >= src_axis_x + src_dims_x.size()) ||
              (dout_dims[ax] > 1 && src_dims_x[ax - src_axis_x] == 1)) {
            reduce_axes_x.push_back(ax);
          } else {
            dst_dims_vec_x.push_back(dout_dims[ax]);
          }
        }
        if (!reduce_axes_x.empty()) {
          tmp_x.Resize(phi::make_ddim(dst_dims_vec_x));
        }
      }
      // dy
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor tmp_y;
      tmp_y.ShareDataWith(*dy);
      if (dy->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec_y;
        std::vector<int> reduce_axes_y;
        auto src_dims_y = dy->dims();
        auto dout_dims = dout->dims();

        int src_axis_y = (src_dims_y.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis_y || ax >= src_axis_y + src_dims_y.size()) ||
              (dout_dims[ax] > 1 && src_dims_y[ax - src_axis_y] == 1)) {
            reduce_axes_y.push_back(ax);
          } else {
            dst_dims_vec_y.push_back(dout_dims[ax]);
          }
        }
        if (!reduce_axes_y.empty()) {
          tmp_y.Resize(phi::make_ddim(dst_dims_vec_y));
        }
      }

      const auto& runner = NpuOpRunner("MinimumGrad",
                                       {*dout, *x, *y},
                                       {tmp_x, tmp_y},
                                       {{"grad_x", true}, {"grad_y", true}});
      runner.Run(stream);

    } else if (dx) {
      Tensor zero_tensor(dout->type());
      zero_tensor.mutable_data<T>(y->dims(), ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&zero_tensor, static_cast<T>(0));
      // dx
      dx->mutable_data<T>(ctx.GetPlace());
      Tensor tmp_x;
      tmp_x.ShareDataWith(*dx);
      if (dx->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec_x;
        std::vector<int> reduce_axes_x;
        auto src_dims_x = dx->dims();
        auto dout_dims = dout->dims();

        int src_axis_x = (src_dims_x.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis_x || ax >= src_axis_x + src_dims_x.size()) ||
              (dout_dims[ax] > 1 && src_dims_x[ax - src_axis_x] == 1)) {
            reduce_axes_x.push_back(ax);
          } else {
            dst_dims_vec_x.push_back(dout_dims[ax]);
          }
        }
        if (!reduce_axes_x.empty()) {
          tmp_x.Resize(phi::make_ddim(dst_dims_vec_x));
        }
      }

      const auto& runner = NpuOpRunner("MinimumGrad",
                                       {*dout, *x, *y},
                                       {tmp_x, zero_tensor},
                                       {{"grad_x", true}, {"grad_y", true}});
      runner.Run(stream);

    } else if (dy) {
      Tensor zero_tensor(dout->type());
      zero_tensor.mutable_data<T>(x->dims(), ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&zero_tensor, static_cast<T>(0));

      // dy
      dy->mutable_data<T>(ctx.GetPlace());
      Tensor tmp_y;
      tmp_y.ShareDataWith(*dy);
      if (dy->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec_y;
        std::vector<int> reduce_axes_y;
        auto src_dims_y = dy->dims();
        auto dout_dims = dout->dims();

        int src_axis_y = (src_dims_y.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis_y || ax >= src_axis_y + src_dims_y.size()) ||
              (dout_dims[ax] > 1 && src_dims_y[ax - src_axis_y] == 1)) {
            reduce_axes_y.push_back(ax);
          } else {
            dst_dims_vec_y.push_back(dout_dims[ax]);
          }
        }
        if (!reduce_axes_y.empty()) {
          tmp_y.Resize(phi::make_ddim(dst_dims_vec_y));
        }
      }

      const auto& runner = NpuOpRunner("MinimumGrad",
                                       {*dout, *x, *y},
                                       {zero_tensor, tmp_y},
                                       {{"grad_x", true}, {"grad_y", true}});
      runner.Run(stream);

    } else {
      std::cout << "error" << std::endl;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    elementwise_min,
    ops::ElementwiseMinNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseMinNPUKernel<paddle::platform::NPUDeviceContext,
                                 paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    elementwise_min_grad,
    ops::ElementwiseMinGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseMinGradNPUKernel<paddle::platform::NPUDeviceContext,
                                     paddle::platform::float16>);
