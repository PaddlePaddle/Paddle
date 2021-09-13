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

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
void PrintTensor(const framework::Tensor& src, const framework::ExecutionContext& ctx){
    std::vector<T> vec(src.numel());
    TensorToVector(src, ctx.device_context(), &vec);
    // for(int i=0; i< static_cast<int>(vec.size()); ++i){
    int len = 10;
    if (len > static_cast<int>(vec.size())) {
      len = static_cast<int>(vec.size());  
    }
    for(int i=0; i< static_cast<int>(10); ++i){
        VLOG(3) << "vec[" << i<< "] : "<< vec[i];
    }
}

template <typename DeviceContext, typename T>
class ElementwiseMulNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*out}, {});
    runner.Run(stream);

    VLOG(3) << "yoki: x: ";
    PrintTensor<T>(*x, ctx);
    VLOG(3) << "yoki: y: ";
    PrintTensor<T>(*y, ctx);
    VLOG(3) << "yoki: out: ";
    PrintTensor<T>(*out, ctx);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMulGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    
    VLOG(3) << "yoki: x: ";
    PrintTensor<T>(*x, ctx);
    VLOG(3) << "yoki: y: ";
    PrintTensor<T>(*y, ctx);
    VLOG(3) << "yoki: dout: ";
    PrintTensor<T>(*dout, ctx);

    if (dx) {
      dx->mutable_data<T>(place);
      const auto& runner_dx = NpuOpRunner("Mul", {*dout, *y}, {*dx}, {});
      runner_dx.Run(stream);
      VLOG(3) << "yoki: dx: ";
      PrintTensor<T>(*dx, ctx);
    }

    if (dy) {
      dy->mutable_data<T>(place);
      const auto& runner_dy = NpuOpRunner("Mul", {*x, *dout}, {*dy}, {});
      runner_dy.Run(stream);
      VLOG(3) << "yoki: dy: ";
      PrintTensor<T>(*dy, ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    elementwise_mul,
    ops::ElementwiseMulNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseMulNPUKernel<paddle::platform::NPUDeviceContext,
                                 paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    elementwise_mul_grad,
    ops::ElementwiseMulGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseMulGradNPUKernel<paddle::platform::NPUDeviceContext,
                                     paddle::platform::float16>);
#endif

