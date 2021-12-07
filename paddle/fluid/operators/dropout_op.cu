/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dropout_impl.cu.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

// It seems that Eigen::Tensor::setRandom in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename Place, typename T>
class GPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* seed =
        context.HasInput("Seed") ? context.Input<Tensor>("Seed") : nullptr;
    auto* y = context.Output<Tensor>("Out");
    y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    bool upscale_in_train = (dropout_implementation == "upscale_in_train");

    bool is_test = context.Attr<bool>("is_test");

    auto& dev_ctx = context.cuda_device_context();
    auto* mask = context.Output<Tensor>("Mask");
    mask->mutable_data<uint8_t>(context.GetPlace());

    bool is_fix_seed = context.Attr<bool>("fix_seed");
    int seed_val = context.Attr<int>("seed");
    DropoutFwGPUKernelDriver<T>(dev_ctx, is_test, dropout_implementation,
                                dropout_prob, upscale_in_train, is_fix_seed,
                                seed_val, *x, seed, mask, y);
  }
};

template <typename DeviceContext, typename T>
class GPUDropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());
    auto size = grad_x->numel();
    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    float dropout_prob = context.Attr<float>("dropout_prob");

    bool is_test = context.Attr<bool>("is_test");

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    DropoutGradGPUKernelDriver<T>(dev_ctx, dropout_implementation, dropout_prob,
                                  *grad_y, *mask, size, grad_x, is_test);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    dropout, ops::GPUDropoutKernel<plat::CUDADeviceContext, float>,
    ops::GPUDropoutKernel<plat::CUDADeviceContext, plat::float16>,
    ops::GPUDropoutKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    dropout_grad, ops::GPUDropoutGradKernel<plat::CUDADeviceContext, float>,
    ops::GPUDropoutGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::GPUDropoutGradKernel<plat::CUDADeviceContext, double>);
