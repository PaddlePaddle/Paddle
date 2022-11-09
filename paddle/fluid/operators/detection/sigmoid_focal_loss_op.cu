/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/detection/sigmoid_focal_loss_op.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/core/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__global__ void GPUSigmoidFocalLossForward(const T *x_data,
                                           const int *label_data,
                                           const int *fg_num_data,
                                           const T gamma,
                                           const T alpha,
                                           const int num_classes,
                                           const int limit,
                                           T *out_data) {
  CUDA_KERNEL_LOOP(i, limit) {
    T x = x_data[i];
    int a = i / num_classes;  // current sample
    int d = i % num_classes;  // current class
    int g = label_data[a];    // target

    // check whether the input data is positive or negative
    // the target classes are in range 1-81
    // and the d is in range 0-80
    T c_pos = static_cast<T>(g == (d + 1));
    T c_neg = static_cast<T>((g != -1) & (g != (d + 1)));

    T fg_num = static_cast<T>((fg_num_data[0] > 1) ? fg_num_data[0] : 1);
    T s_neg = (1.0 - alpha) / fg_num;
    T s_pos = alpha / fg_num;

    // p = 1. / 1. + expf(-x)
    T p = 1. / (1. + real_exp(-x));

    // (1 - p)**gamma * log(p)
    T term_pos = std::pow(static_cast<T>(1. - p), gamma) *
                 real_log(p > FLT_MIN ? p : FLT_MIN);
    // p**gamma * log(1 - p)
    T term_neg =
        std::pow(p, gamma) *
        (-1. * x * (x >= 0) - real_log(1. + real_exp(x - 2. * x * (x >= 0))));

    out_data[i] = 0.0;
    out_data[i] += -c_pos * term_pos * s_pos;
    out_data[i] += -c_neg * term_neg * s_neg;
  }
}

template <typename T>
__global__ void GPUSigmoidFocalLossBackward(const T *x_data,
                                            const int *label_data,
                                            const int *fg_num_data,
                                            const T gamma,
                                            const T alpha,
                                            const int num_classes,
                                            const T *dout_data,
                                            const int limit,
                                            T *dx_data) {
  CUDA_KERNEL_LOOP(i, limit) {
    T x = x_data[i];
    T dout = dout_data[i];

    int a = i / num_classes;  // current sample
    int d = i % num_classes;  // current class

    T fg_num = static_cast<T>((fg_num_data[0] > 1) ? fg_num_data[0] : 1);
    T s_neg = (1.0 - alpha) / fg_num;
    T s_pos = alpha / fg_num;

    int g = label_data[a];
    T c_pos = static_cast<T>(g == (d + 1));
    T c_neg = static_cast<T>((g != -1) & (g != (d + 1)));

    T p = 1. / (1. + real_exp(-x));

    // (1-p)**g * (1 - p - g*p*log(p))
    T term_pos = std::pow(static_cast<T>(1. - p), gamma) *
                 (1. - p - (p * gamma * real_log(p > FLT_MIN ? p : FLT_MIN)));
    // (p**g) * (g*(1-p)*log(1-p) - p)
    T term_neg =
        std::pow(p, gamma) *
        ((-1. * x * (x >= 0) - real_log(1. + real_exp(x - 2. * x * (x >= 0)))) *
             (1. - p) * gamma -
         p);

    dx_data[i] = 0.0;
    dx_data[i] += -c_pos * s_pos * term_pos;
    dx_data[i] += -c_neg * s_neg * term_neg;
    dx_data[i] = dx_data[i] * dout;
  }
}

template <typename DeviceContext, typename T>
class GPUSigmoidFocalLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<phi::DenseTensor>("X");
    const Tensor *Labels = context.Input<phi::DenseTensor>("Label");
    const Tensor *FgNum = context.Input<phi::DenseTensor>("FgNum");
    Tensor *Out = context.Output<phi::DenseTensor>("Out");
    T gamma = static_cast<T>(context.Attr<float>("gamma"));
    T alpha = static_cast<T>(context.Attr<float>("alpha"));
    auto x_dims = X->dims();
    int num_classes = static_cast<int>(x_dims[1]);
    auto out_data = Out->mutable_data<T>(context.GetPlace());

    auto &dev_ctx = context.cuda_device_context();

    int limit = Out->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;
    GPUSigmoidFocalLossForward<T>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(X->data<T>(),
                                                   Labels->data<int>(),
                                                   FgNum->data<int>(),
                                                   gamma,
                                                   alpha,
                                                   num_classes,
                                                   limit,
                                                   out_data);
  }
};

template <typename DeviceContext, typename T>
class GPUSigmoidFocalLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<phi::DenseTensor>("X");
    const Tensor *Labels = context.Input<phi::DenseTensor>("Label");
    const Tensor *FgNum = context.Input<phi::DenseTensor>("FgNum");
    const Tensor *dOut =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    Tensor *dX = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto dx_data = dX->mutable_data<T>(context.GetPlace());
    T gamma = static_cast<T>(context.Attr<float>("gamma"));
    T alpha = static_cast<T>(context.Attr<float>("alpha"));
    auto x_dims = X->dims();
    int num_classes = static_cast<int>(x_dims[1]);

    auto &dev_ctx = context.cuda_device_context();

    int limit = dX->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;
    GPUSigmoidFocalLossBackward<T>
        <<<blocks, threads, 0, dev_ctx.stream()>>>(X->data<T>(),
                                                   Labels->data<int>(),
                                                   FgNum->data<int>(),
                                                   gamma,
                                                   alpha,
                                                   num_classes,
                                                   dOut->data<T>(),
                                                   limit,
                                                   dx_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sigmoid_focal_loss,
    ops::GPUSigmoidFocalLossKernel<phi::GPUContext, float>,
    ops::GPUSigmoidFocalLossKernel<phi::GPUContext, double>);
REGISTER_OP_CUDA_KERNEL(
    sigmoid_focal_loss_grad,
    ops::GPUSigmoidFocalLossGradKernel<phi::GPUContext, float>,
    ops::GPUSigmoidFocalLossGradKernel<phi::GPUContext, double>);
