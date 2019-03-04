/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx = context.template device_context<platform::CUDADeviceContext>();
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");
    // auto dims = X->dims();
    const int axis = context.Attr<int>("axis");
    int rank = X->dims().size();

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());

    std::vector<int> perm, shape;
    CalcTransPermAndShapeByAxis(*X, axis, &perm, &shape);

    Tensor X_2d, Out_2d;
    Tensor X_trans, Out_trans;
    if (axis != -1 && axis != rank - 1) {
      X_trans.mutable_data<T>(framework::make_ddim(shape), context.GetPlace());
      Out_trans.mutable_data<T>(framework::make_ddim(shape), context.GetPlace());
      TransCompute<platform::CUDADeviceContext, T>(rank, dev_ctx, *X, &X_trans, perm);
      TransCompute<platform::CUDADeviceContext, T>(rank, dev_ctx, *Out, &Out_trans, perm);
      X_2d = framework::ReshapeToMatrix(X_trans, rank - 1);
      Out_2d = framework::ReshapeToMatrix(Out_trans, rank - 1);
    } else {
      X_2d = framework::ReshapeToMatrix(*X, rank - 1);
      Out_2d = framework::ReshapeToMatrix(*Out, rank - 1);
    }

    math::SoftmaxCUDNNFunctor<T>()(
        context.template device_context<platform::CUDADeviceContext>(),
        &X_2d, &Out_2d);

    if (axis != -1 && axis != rank - 1) {
      TransCompute<platform::CUDADeviceContext, T>(rank, dev_ctx, Out_trans, Out, perm);
    }
  }
};

template <typename T>
class SoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx = context.template device_context<platform::CUDADeviceContext>();
    auto* Out = context.Input<Tensor>("Out");
    auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
    const int axis = context.Attr<int>("axis");
    int rank = Out->dims().size();

    // allocate memory on device.
    dX->mutable_data<T>(context.GetPlace());

    std::vector<int> perm, shape;
    CalcTransPermAndShapeByAxis(*dX, axis, &perm, &shape);

    Tensor dX_2d, Out_2d, dOut_2d;
    Tensor dX_trans, Out_trans, dOut_trans;
    if (axis != -1 && axis != rank - 1) {
      dX_trans.mutable_data<T>(framework::make_ddim(shape), context.GetPlace());
      Out_trans.mutable_data<T>(framework::make_ddim(shape), context.GetPlace());
      dOut_trans.mutable_data<T>(framework::make_ddim(shape), context.GetPlace());
      TransCompute<platform::CUDADeviceContext, T>(rank, dev_ctx, *dX, &dX_trans, perm);
      TransCompute<platform::CUDADeviceContext, T>(rank, dev_ctx, *Out, &Out_trans, perm);
      TransCompute<platform::CUDADeviceContext, T>(rank, dev_ctx, *dOut, &dOut_trans, perm);
      dX_2d = framework::ReshapeToMatrix(dX_trans, rank - 1);
      Out_2d = framework::ReshapeToMatrix(Out_trans, rank - 1);
      dOut_2d = framework::ReshapeToMatrix(dOut_trans, rank - 1);
    } else {
      dX_2d = framework::ReshapeToMatrix(*dX, rank - 1);
      Out_2d = framework::ReshapeToMatrix(*Out, rank - 1);
      dOut_2d = framework::ReshapeToMatrix(*dOut, rank - 1);
    }

    math::SoftmaxGradCUDNNFunctor<T>()(
        context.template device_context<platform::CUDADeviceContext>(),
        &Out_2d, &dOut_2d, &dX_2d);

    if (axis != -1 && axis != rank - 1) {
      TransCompute<platform::CUDADeviceContext, T>(rank, dev_ctx, dX_trans, dX, perm);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
