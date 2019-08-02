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
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/dropout_op_cache.h"
#include "paddle/fluid/platform/cudnn_desc.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using TensorDescriptor = platform::TensorDescriptor;
using DataLayout = platform::DataLayout;

template <typename T>
class CUDNNDropoutOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use CUDAPlace.");
    const Tensor *x = ctx.Input<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");
    Tensor *mask = ctx.Output<Tensor>("Mask");

    const T *x_data = x->data<T>();
    T *out_data = out->mutable_data<T>(ctx.GetPlace());

    auto size_prod = x->numel();
    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_test = ctx.Attr<bool>("is_test");
    if (is_test) {
      TensorCopySync(*x, ctx.GetPlace(), out);
      return;
    }

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto *cache_var = ctx.InputVar("Cache");
    if (!cache_var) {
      // The RAW type cache variable wouldn't be created and broadcasted on
      // multi-devices before the first running.
      // use parent scope to make cache persistable
      auto *scope = const_cast<framework::Scope *>(ctx.scope().parent());
      auto cache_var_name = ctx.InputVarName("Cache");
      cache_var = scope->Var(cache_var_name);
    }
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
    auto &cudnn_data_desc = cudnn_dropout_cache->data_desc_;
    auto &reserve_space_size_in_bytes =
        cudnn_dropout_cache->reserve_space_size_in_bytes_;
    if (size_prod != cudnn_dropout_cache->input_size_) {
      cudnn_dropout_cache->input_size_ = size_prod;
      CUDNN_ENFORCE(platform::dynload::cudnnSetTensor4dDescriptor(
          cudnn_data_desc, GetCudnnTensorFormat(DataLayout::kNCHW),
          paddle::platform::CudnnDataType<T>::type, size_prod, 1, 1, 1));
      CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetReserveSpaceSize(
          cudnn_data_desc, &reserve_space_size_in_bytes));
    }
    auto &cudnn_dropout_desc = cudnn_dropout_cache->dropout_desc_;
    mask->Resize({static_cast<int64_t>(reserve_space_size_in_bytes)});
    CUDNN_ENFORCE(platform::dynload::cudnnDropoutForward(
        handle, cudnn_dropout_desc, cudnn_data_desc, x_data, cudnn_data_desc,
        out_data, mask->mutable_data<uint8_t>(ctx.GetPlace()),
        reserve_space_size_in_bytes));
  }
};

template <typename T>
class CUDNNDropoutGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    const Tensor *mask = ctx.Input<Tensor>("Mask");
    Tensor *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    const T *out_grad_data = out_grad->data<T>();
    T *x_grad_data = x_grad->mutable_data<T>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto *cache_var = ctx.InputVar("Cache");
    PADDLE_ENFORCE(cache_var->IsInitialized());
    CudnnDropoutCache *cudnn_dropout_cache =
        const_cast<framework::Variable *>(cache_var)
            ->GetMutable<CudnnDropoutCache>();

    auto &cudnn_data_desc = cudnn_dropout_cache->data_desc_;
    auto &cudnn_dropout_desc = cudnn_dropout_cache->dropout_desc_;
    auto &reserve_space_size_in_bytes =
        cudnn_dropout_cache->reserve_space_size_in_bytes_;
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
REGISTER_OP_KERNEL(dropout, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNDropoutOpKernel<float>,
                   ops::CUDNNDropoutOpKernel<double>);
REGISTER_OP_KERNEL(dropout_grad, CUDNN, ::paddle::platform::CUDAPlace,
                   ops::CUDNNDropoutGradOpKernel<float>,
                   ops::CUDNNDropoutGradOpKernel<double>);
