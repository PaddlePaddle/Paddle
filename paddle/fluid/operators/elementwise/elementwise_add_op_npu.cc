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

template <typename T>
class ElementwiseAddNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    int axis = ctx.Attr<int>("axis");

    bool direct_compute = false;
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

    if (x_dims.size() == y_dims.size()) {
      direct_compute = true;
    } else if (x_dims.size() > y_dims.size()) {
      direct_compute = x_dims.size() == (y_dims.size() + axis);
    } else {
      direct_compute = y_dims.size() == (x_dims.size() + axis);
    }

    if (direct_compute) {
      const auto& runner = NpuOpRunner("Add", {*x, *y}, {*out}, {});
      runner.Run(dev_ctx.stream());
    } else {
      Tensor transformed_x, transformed_y;
      NpuElementWiseOpBroadcast<T>(
          dev_ctx, x, y, axis, &transformed_x, &transformed_y);
      const auto& runner =
          NpuOpRunner("Add", {transformed_x, transformed_y}, {*out}, {});
      runner.Run(dev_ctx.stream());
    }
  }
};

template <typename T>
class ElementwiseAddGradNPUKernel : public framework::OpKernel<T> {
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
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      if (dx->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec;
        std::vector<int> reduce_axes;
        auto src_dims = dx->dims();
        auto dout_dims = dout->dims();

        int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
              (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          } else {
            dst_dims_vec.push_back(dout_dims[ax]);
          }
        }
        if (!reduce_axes.empty()) {
          Tensor tmp;
          tmp.ShareDataWith(*dx);
          tmp.Resize(phi::make_ddim(dst_dims_vec));
          const auto& runner =
              NpuOpRunner("ReduceSumD",
                          {*dout},
                          {tmp},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        framework::TensorCopy(*dout, ctx.GetPlace(), dev_ctx, dx);
      }
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
      if (dy->dims() != dout->dims()) {
        std::vector<int> dst_dims_vec;
        std::vector<int> reduce_axes;
        auto src_dims = dy->dims();
        auto dout_dims = dout->dims();

        int src_axis = (src_dims.size() < dout_dims.size() ? axis : 0);
        for (int ax = 0; ax < dout_dims.size(); ++ax) {
          if ((ax < src_axis || ax >= src_axis + src_dims.size()) ||
              (dout_dims[ax] > 1 && src_dims[ax - src_axis] == 1)) {
            reduce_axes.push_back(ax);
          } else {
            dst_dims_vec.push_back(dout_dims[ax]);
          }
        }
        if (!reduce_axes.empty()) {
          Tensor tmp;
          tmp.ShareDataWith(*dy);
          tmp.Resize(phi::make_ddim(dst_dims_vec));
          const auto& runner =
              NpuOpRunner("ReduceSumD",
                          {*dout},
                          {tmp},
                          {{"axes", reduce_axes}, {"keep_dims", false}});
          runner.Run(stream);
        }
      } else {
        framework::TensorCopy(*dout, ctx.GetPlace(), dev_ctx, dy);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(elementwise_add,
                       ops::ElementwiseAddNPUKernel<float>,
#ifdef PADDLE_WITH_ASCEND_INT64
                       ops::ElementwiseAddNPUKernel<int64_t>,
#endif
                       ops::ElementwiseAddNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(elementwise_add_grad,
                       ops::ElementwiseAddGradNPUKernel<float>,
                       ops::ElementwiseAddGradNPUKernel<plat::float16>);
