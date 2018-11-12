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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cudnn_helper.h"

#if CUDNN_VERSION >= 7001

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedDropoutDescriptor = platform::ScopedDropoutDescriptor;
using DataLayout = platform::DataLayout;

template <typename T>
class DropoutCUDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *x = ctx.Input<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");
    Tensor *mask = ctx.Output<Tensor>("Mask");
    Tensor *cache = const_cast<Tensor *>(ctx.Input<Tensor>("Cache"));

    const T *x_data = x->data<T>();
    T *out_data = out->mutable_data<T>(ctx.GetPlace());

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_test = ctx.Attr<bool>("is_test");
    if (is_test) {
      TensorCopy(*x, ctx.GetPlace(), out);
      return;
    }
    std::random_device rnd;
    int seed = ctx.Attr<bool>("fix_seed") ? ctx.Attr<int>("seed") : rnd();

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    ScopedDropoutDescriptor dropout_desc;
    size_t state_size_in_bytes = 0;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetStatesSize(
        handle, &state_size_in_bytes));  // may also cache this
    auto cudnn_dropout_desc = dropout_desc.descriptor(
        handle, dropout_prob, cache->mutable_data<uint8_t>(ctx.GetPlace()),
        state_size_in_bytes, static_cast<uint64_t>(seed),
        cache->IsInitialized());
    ScopedTensorDescriptor data_desc;
    auto cudnn_data_desc =
        data_desc.descriptor<T>(DataLayout::kNCHW, x->numel(), 1, 1, 1);
    size_t reserve_space_size_in_bytes = 0;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetReserveSpaceSize(
        cudnn_data_desc, &reserve_space_size_in_bytes));
    mask->Resize({static_cast<int64_t>(reserve_space_size_in_bytes)});
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutForward(
        handle, cudnn_dropout_desc, cudnn_data_desc, x_data, cudnn_data_desc,
        out_data, mask->mutable_data<uint8_t>(ctx.GetPlace()),
        reserve_space_size_in_bytes));
  }
};

template <typename T>
class DropoutCUDNNGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    const Tensor *mask = ctx.Input<Tensor>("Mask");
    const Tensor *cache = ctx.Input<Tensor>("Cache");

    const T *out_grad_data = out_grad->data<T>();
    T *x_grad_data = x_grad->mutable_data<T>(ctx.GetPlace());

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    std::random_device rnd;
    int seed = ctx.Attr<bool>("fix_seed") ? ctx.Attr<int>("seed") : rnd();

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    ScopedDropoutDescriptor dropout_desc;
    size_t state_size_in_bytes = 0;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetStatesSize(
        handle, &state_size_in_bytes));  // may also cache this
    auto cudnn_dropout_desc = dropout_desc.descriptor(
        handle, dropout_prob, const_cast<uint8_t *>(cache->data<uint8_t>()),
        state_size_in_bytes, static_cast<uint64_t>(seed),
        cache->IsInitialized());
    ScopedTensorDescriptor data_desc;
    auto cudnn_data_desc =
        data_desc.descriptor<T>(DataLayout::kNCHW, out_grad->numel(), 1, 1, 1);
    size_t reserve_space_size_in_bytes = 0;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetReserveSpaceSize(
        cudnn_data_desc, &reserve_space_size_in_bytes));
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutBackward(
        handle, cudnn_dropout_desc, cudnn_data_desc, out_grad_data,
        cudnn_data_desc, x_grad_data,
        const_cast<uint8_t *>(mask->data<uint8_t>()),
        reserve_space_size_in_bytes));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_KERNEL(dropout, CUDNN, plat::CUDAPlace,
                   ops::DropoutCUDNNOpKernel<float>,
                   ops::DropoutCUDNNOpKernel<double>);
REGISTER_OP_KERNEL(dropout_grad, CUDNN, plat::CUDAPlace,
                   ops::DropoutCUDNNGradOpKernel<float>,
                   ops::DropoutCUDNNGradOpKernel<double>);

#endif
