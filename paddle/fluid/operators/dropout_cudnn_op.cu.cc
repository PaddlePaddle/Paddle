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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = platform::DataLayout;

struct CudnnDropoutCache {
  CudnnDropoutCache() {
    PADDLE_ENFORCE(
        platform::dynload::cudnnCreateDropoutDescriptor(&dropout_desc_));
  }
  ~CudnnDropoutCache() {
    PADDLE_ENFORCE(
        platform::dynload::cudnnDestroyDropoutDescriptor(dropout_desc_));
  }
  // cudnnSetDropoutDescriptor needs expensive precomputation to initialize
  // the random number generator states, so cache the states.
  cudnnDropoutDescriptor_t dropout_desc_;
  size_t states_size_in_bytes_;
  Tensor states_;  // dtype = uint8_t
};

template <typename T>
class DropoutCUDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *x = ctx.Input<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");
    Tensor *mask = ctx.Output<Tensor>("Mask");
    const T *x_data = x->data<T>();
    T *out_data = out->mutable_data<T>(ctx.GetPlace());

    auto size_prod = x->numel();
    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_test = ctx.Attr<bool>("is_test");
    if (is_test) {
      return;
    }

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto *cache_var = ctx.InputVar("Cache");
    CudnnDropoutCache *cudnn_dropout_cache = nullptr;
    if (cache_var->IsInitialized()) {
      cudnn_dropout_cache = const_cast<framework::Variable *>(cache_var)
                                ->GetMutable<CudnnDropoutCache>();
    } else {
      cudnn_dropout_cache = const_cast<framework::Variable *>(cache_var)
                                ->GetMutable<CudnnDropoutCache>();
      auto &states_size_in_bytes = cudnn_dropout_cache->states_size_in_bytes_;
      CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetStatesSize(
          handle, &states_size_in_bytes));
      auto &states = cudnn_dropout_cache->states_;
      states.Resize({static_cast<int64_t>(states_size_in_bytes)});
      auto *states_data = states.mutable_data<uint8_t>(ctx.GetPlace());
      std::random_device rnd;
      int seed = ctx.Attr<bool>("fix_seed") ? ctx.Attr<int>("seed") : rnd();
      CUDNN_ENFORCE(platform::dynload::cudnnSetDropoutDescriptor(
          cudnn_dropout_cache->dropout_desc_, handle, dropout_prob, states_data,
          states_size_in_bytes, static_cast<uint64_t>(seed)));
    }

    ScopedTensorDescriptor data_desc;
    auto &cudnn_data_desc =
        data_desc.descriptor<T>(DataLayout::kNCHW, size_prod, 1, 1, 1);
    size_t reserve_space_size_in_bytes = 0;
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetReserveSpaceSize(
        cudnn_data_desc, &reserve_space_size_in_bytes));
    mask->Resize({static_cast<int64_t>(reserve_space_size_in_bytes)});
    auto &cudnn_dropout_desc = cudnn_dropout_cache->dropout_desc_;
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
    // const Tensor *grad_out =
    // ctx.Input<Tensor>(framework::GradVarName("Out"));
    // Tensor *grad_x = ctx.Output<Tensor>(framework::GradVarName("X"));

    // const T *grad_out_data = grad_out->data<T>();
    // T *grad_x_data = grad_x->mutable_data<T>(ctx.GetPlace());

    // auto &dev_ctx = ctx.template
    // device_context<platform::CUDADeviceContext>();
    // auto handle = dev_ctx.cudnn_handle();
    // auto *cache_var = ctx.InputVar("Cache");
    // PADDLE_ENFORCE(cache_var->IsInitialized());
    // CudnnDropoutCache *cudnn_dropout_cache =
    //     const_cast<framework::Variable *>(cache_var)
    //         ->GetMutable<CudnnDropoutCache>();

    // auto &cudnn_data_desc = cudnn_dropout_cache->data_desc_;
    // auto &cudnn_dropout_desc = cudnn_dropout_cache->dropout_desc_;
    // auto &mask = cudnn_dropout_cache->mask_;
    // auto *mask_data = mask.data<uint8_t>();
    // auto &reserve_space_size_in_bytes =
    //     cudnn_dropout_cache->reserve_space_size_in_bytes_;
    // CUDNN_ENFORCE(platform::dynload::cudnnDropoutBackward(
    //     handle, cudnn_dropout_desc, cudnn_data_desc, grad_out_data,
    //     cudnn_data_desc, grad_x_data, mask_data,
    //     reserve_space_size_in_bytes));
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
                   ops::DropoutCUDNNOpKernel<double>);
