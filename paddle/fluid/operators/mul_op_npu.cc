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

#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    PADDLE_ENFORCE_EQ(x_num_col_dims, 1,
                      platform::errors::InvalidArgument(
                          "now only support x_num_col_dims == 1: but got %d",
                          x_num_col_dims));
    PADDLE_ENFORCE_EQ(y_num_col_dims, 1,
                      platform::errors::InvalidArgument(
                          "now only support y_num_col_dims == 1: but got %d",
                          y_num_col_dims));
    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x->dims().size() == 2 && y->dims().size() == 2) {
        out->mutable_data<T>(ctx.GetPlace());
        auto runner =
            NpuOpRunner("MatMul", {*x, *y}, {*out},
                        {{"transpose_x1", false}, {"transpose_x2", false}});

        runner.Run(stream);
      } else if (x->dims().size() == 3 && y->dims().size() == 2) {
        // flatten
        Tensor tmp_flatten(x->type());
        int64_t size = x->dims()[1] * x->dims()[2];
        x->resize(x->dims()[0], size)
            // std::vector<int64_t> vec_flatten;
            // vec_flatten.push_back(size);
            // tmp_flatten.Resize(framework::make_ddim(vec_flatten));
            // tmp_flatten.mutable_data<T>(ctx.GetPlace());
            // auto runner_flatten = NpuOpRunner("Flatten", {*x}, {tmp_flatten},
            // {});
            // runner_flatten.Run(stream);
            out->mutable_data<T>(ctx.GetPlace());
        // matmul
        auto runner_matmul =
            NpuOpRunner("MatMul", x, *y
      }, {*out}, {});
      runner_matmul.Run(stream);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument("not suppert dims"));
    }
    // to do other
  }
};

template <typename DeviceContext, typename T>
class MulGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    if (x_num_col_dims == 1 && y_num_col_dims == 1) {
      if (x->dims().size() == 2 && y->dims().size() == 2) {
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          auto runner_dx =
              NpuOpRunner("MatMul", {*dout, *y}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", true}});

          runner_dx.Run(stream);
        }

        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          auto runner_dy =
              NpuOpRunner("MatMul", {*x, *dout}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
        }
      } else if (x->dims().size() == 3 && y->dims().size() == 2) {
        // flatten
        Tensor tmp_flatten(x->type());
        int64_t size = x->dims()[1] * x->dims()[2];
        std::vector<int64_t> vec_flatten;
        vec_flatten.push_back(size);
        tmp_flatten.Resize(framework::make_ddim(vec_flatten));
        tmp_flatten.mutable_data<T>(ctx.GetPlace());
        auto runner_flatten = NpuOpRunner("Flatten", {*x}, {tmp_flatten}, {});
        runner_flatten.Run(stream);
        // matmul
        if (dx) {
          dx->mutable_data<T>(ctx.GetPlace());
          auto runner_dx =
              NpuOpRunner("MatMul", {*dout, *y}, {*dx},
                          {{"transpose_x1", false}, {"transpose_x2", true}});

          runner_dx.Run(stream);
        }
        // to do shape==2

        if (dy) {
          dy->mutable_data<T>(ctx.GetPlace());
          auto runner_dy =
              NpuOpRunner("MatMul", {tmp_flatten, *dout}, {*dy},
                          {{"transpose_x1", true}, {"transpose_x2", false}});

          runner_dy.Run(stream);
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    mul, ops::MulNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MulNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);
REGISTER_OP_NPU_KERNEL(
    mul_grad, ops::MulGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MulGradNPUKernel<paddle::platform::NPUDeviceContext,
                          paddle::platform::float16>);
