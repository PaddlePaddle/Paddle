/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/cross_entropy_op.h"

namespace paddle {
namespace operators {

namespace {

template <typename T>
__global__ void CrossEntropyGradientKernel(T* dX, const T* dY, const T* X,
                                           const int64_t* label, const int N,
                                           const int D) {
  // TOOD(qingqing) define CUDA_1D_KERNEL_LOOP macro in a common file.
  // CUDA_1D_KERNEL_LOOP(i, N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    int idx = i * D + label[i];
    dX[idx] = -dY[i] / X[idx];
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* dX, const T* dY, const T* X,
                                               const T* label, const int N,
                                               const int D) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < N * D) {
    int row_ids = ids / D;
    dX[ids] = -label[ids] * dY[row_ids] / X[ids];
  }
}
}  // namespace

template <typename T>
class CrossEntropyOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* label = ctx.Input<Tensor>("Label");
    Tensor* y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    math::CrossEntropyFunctor<platform::GPUPlace, T>()(
        ctx.device_context(), y, x, label, ctx.Attr<bool>("soft_label"));
  }
};

template <typename T>
class CrossEntropyGradientOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");

    const Tensor* x = ctx.Input<Tensor>("X");
    const Tensor* label = ctx.Input<Tensor>("Label");
    Tensor* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    const T* dy_data =
        ctx.Input<Tensor>(framework::GradVarName("Y"))->data<T>();
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());
    const T* x_data = x->data<T>();

    int64_t batch_size = x->dims()[0];
    int64_t class_num = x->dims()[1];

    int block = 512;
    int grid = (batch_size * class_num + block - 1) / block;
    auto stream = ctx.cuda_device_context().stream();

    if (ctx.Attr<bool>("soft_label")) {
      auto* label_data = label->data<T>();
      SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          dx_data, dy_data, x_data, label_data, batch_size, class_num);
    } else {
      math::SetConstant<platform::GPUPlace, T> functor;
      functor(ctx.device_context(), dx, 0);
      auto* label_data = label->data<int64_t>();
      grid = (batch_size + block - 1) / block;
      CrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          dx_data, dy_data, x_data, label_data, batch_size, class_num);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(cross_entropy, ops::CrossEntropyOpCUDAKernel<float>,
                       ops::CrossEntropyOpCUDAKernel<double>);
REGISTER_OP_GPU_KERNEL(cross_entropy_grad,
                       ops::CrossEntropyGradientOpCUDAKernel<float>,
                       ops::CrossEntropyGradientOpCUDAKernel<double>);
