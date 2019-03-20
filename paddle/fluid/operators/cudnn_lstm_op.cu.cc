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
#include "paddle/fluid/operators/cudnn_rnn_cache.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename T>
class CudnnLSTMGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *x = ctx.Input<Tensor>("Input");
    const Tensor *init_h = ctx.Input<Tensor>("InitH");
    const Tensor *init_c = ctx.Input<Tensor>("InitC");

    auto w = ctx.Input<Tensor>("W");

    Tensor *out = ctx.Output<Tensor>("Out");
    Tensor *last_h = ctx.Output<Tensor>("last_h");
    Tensor *last_c = ctx.Output<Tensor>("last_c");

    const T *x_data = x->data<T>();
    const T *init_h_data = init_h->data<T>();
    const T *init_c_data = init_c->data<T>();

    const T *w_data = w->data<T>();

    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    T *last_h_data = last_h->mutable_data<T>(ctx.GetPlace());
    T *last_c_data = last_c->mutable_data<T>(ctx.GetPlace());

    size_t max_len = ctx.Attr<int>("max_len");
    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int input_size = ctx.Attr<int>("input_size");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    bool is_test = ctx.Attr<bool>("is_test");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto *cache_var = ctx.InputVar("Cache");
    if (!cache_var) {
      // The RAW type cache variable wouldn't be created and broadcasted on
      // multi-devices before the first running.
      // use parent scope to make cache persistable
      auto *scope = const_cast<framework::Scope *>(ctx.scope().parent());
      auto cache_var_name = ctx.Inputs("Cache")[0];
      cache_var = scope->Var(cache_var_name);
    }
    CudnnRNNCache *cudnn_rnn_cache = nullptr;
    if (cache_var->IsInitialized()) {
      // const_cast is usually bad.
      cudnn_rnn_cache = const_cast<framework::Variable *>(cache_var)
                            ->GetMutable<CudnnRNNCache>();
    } else {
      // const_cast is usually bad.
      cudnn_rnn_cache = const_cast<framework::Variable *>(cache_var)
                            ->GetMutable<CudnnRNNCache>();
      std::random_device rnd;
      int seed = ctx.Attr<int>("seed");
      if (seed == -1) {
        seed = rnd();
      }

      auto input_w_numel = w->numel();
      auto batch_size = x->dims()[1];
      cudnn_rnn_cache->init(handle, ctx.GetPlace(), max_len, batch_size,
                            input_size, hidden_size, num_layers, dropout_prob,
                            is_bidirec, seed, input_w_numel);
    }

    auto run_seq_len = x->dims()[0];

    if (is_test) {
      // for inference
      CUDNN_ENFORCE(platform::dynload::cudnnRNNForwardInference(
          handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
          cudnn_rnn_cache->x_desc_, x_data, cudnn_rnn_cache->hx_desc_,
          init_h_data, cudnn_rnn_cache->cx_desc_, init_c_data,
          cudnn_rnn_cache->w_desc_, w_data, cudnn_rnn_cache->y_desc_, out_data,
          cudnn_rnn_cache->hy_desc_, last_h_data, cudnn_rnn_cache->cy_desc_,
          last_c_data, cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
          cudnn_rnn_cache->workspace_size_));
    } else {
      // for train
      CUDNN_ENFORCE(platform::dynload::cudnnRNNForwardTraining(
          handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
          cudnn_rnn_cache->x_desc_, x_data, cudnn_rnn_cache->hx_desc_,
          init_h_data, cudnn_rnn_cache->cx_desc_, init_c_data,
          cudnn_rnn_cache->w_desc_, w_data, cudnn_rnn_cache->y_desc_, out_data,
          cudnn_rnn_cache->hy_desc_, last_h_data, cudnn_rnn_cache->cy_desc_,
          last_c_data, cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
          cudnn_rnn_cache->workspace_size_,
          cudnn_rnn_cache->reserve_data_.data<uint8_t>(),
          cudnn_rnn_cache->reserve_size_));
    }
  }
};

template <typename T>
class CudnnLSTMGPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input = ctx.Input<Tensor>("Input");
    auto *weight = ctx.Input<Tensor>("W");
    auto *init_h = ctx.Input<Tensor>("InitH");
    auto *init_c = ctx.Input<Tensor>("InitC");
    // auto * last_h = ctx.Input<Tensor>("last_h");
    // auto * last_c = ctx.Input<Tensor>("last_c");
    auto *out = ctx.Input<Tensor>("Out");
    auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *last_h_grad = ctx.Input<Tensor>(framework::GradVarName("last_h"));
    auto *last_c_grad = ctx.Input<Tensor>(framework::GradVarName("last_c"));

    // auto* init_h = ctx.Input<Tensor>("init_h");
    // auto* init_c = ctx.Input<Tensor>("init_c");

    auto *in_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto *weight_grad = ctx.Output<Tensor>(framework::GradVarName("W"));
    auto *init_h_grad = ctx.Output<Tensor>(framework::GradVarName("InitH"));
    auto *init_c_grad = ctx.Output<Tensor>(framework::GradVarName("InitC"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto *cache_var = ctx.InputVar("Cache");
    PADDLE_ENFORCE(cache_var->IsInitialized());
    CudnnRNNCache *cudnn_rnn_cache =
        const_cast<framework::Variable *>(cache_var)
            ->GetMutable<CudnnRNNCache>();

    auto input_dims = input->dims();
    auto init_h_dims = init_h->dims();
    auto init_c_dims = init_c->dims();
    in_grad->mutable_data<T>(ctx.GetPlace());
    weight_grad->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<paddle::platform::CUDADeviceContext, T> zero;
    zero(dev_ctx, in_grad, static_cast<T>(0.0));
    zero(dev_ctx, weight_grad, static_cast<T>(0.0));

    T *init_h_grad_data = NULL;
    if (init_h_grad == nullptr) {
      Tensor init_h_grad_temp;
      init_h_grad_temp.mutable_data<T>(init_h_dims, ctx.GetPlace());
      zero(dev_ctx, &init_h_grad_temp, static_cast<T>(0.0));

      init_h_grad_data = init_h_grad_temp.data<T>();
    } else {
      init_h_grad->mutable_data<T>(init_h_dims, ctx.GetPlace());
      zero(dev_ctx, init_h_grad, static_cast<T>(0.0));
      init_h_grad_data = init_h_grad->data<T>();
    }

    T *init_c_grad_data = NULL;
    if (init_c_grad == nullptr) {
      Tensor init_c_grad_temp;
      init_c_grad_temp.mutable_data<T>(init_c_dims, ctx.GetPlace());
      zero(dev_ctx, &init_c_grad_temp, static_cast<T>(0.0));

      init_c_grad_data = init_c_grad_temp.data<T>();
    } else {
      init_c_grad->mutable_data<T>(init_c_dims, ctx.GetPlace());
      zero(dev_ctx, init_c_grad, static_cast<T>(0.0));
      init_c_grad_data = init_c_grad->data<T>();
    }

    const T *last_h_grad_data = NULL;
    if (last_h_grad == nullptr) {
      Tensor last_h_grad_temp;
      last_h_grad_temp.mutable_data<T>(init_h_dims, ctx.GetPlace());
      zero(dev_ctx, &last_h_grad_temp, static_cast<T>(0.0));

      last_h_grad_data = (const T *)last_h_grad_temp.data<T>();
    } else {
      last_h_grad_data = last_h_grad->data<T>();
    }

    const T *last_c_grad_data = NULL;
    if (last_c_grad == nullptr) {
      Tensor last_c_grad_temp;
      last_c_grad_temp.mutable_data<T>(init_c_dims, ctx.GetPlace());
      zero(dev_ctx, &last_c_grad_temp, static_cast<T>(0.0));

      last_c_grad_data = (const T *)last_c_grad_temp.data<T>();
    } else {
      last_c_grad_data = last_c_grad->data<T>();
    }

    const T *out_grad_data = NULL;
    if (out_grad == nullptr) {
      Tensor out_grad_temp;
      out_grad_temp.mutable_data<T>(out->dims(), ctx.GetPlace());
      zero(dev_ctx, &out_grad_temp, static_cast<T>(0.0));

      out_grad_data = (const T *)out_grad_temp.data<T>();
    } else {
      out_grad_data = out_grad->data<T>();
    }

    // zero( dev_ctx, last_h_grad, static_cast<T>(0.0));
    // zero( dev_ctx, last_c_grad, static_cast<T>(0.0));

    auto out_data = out->data<T>();
    // auto out_grad_data = out_grad->data<T>();
    auto weight_data = weight->data<T>();
    auto init_h_data = init_h->data<T>();
    auto init_c_data = init_c->data<T>();
    auto in_grad_data = in_grad->data<T>();

    auto work_data = cudnn_rnn_cache->workspace_data_.data<uint8_t>();
    auto reserve_data = cudnn_rnn_cache->reserve_data_.data<uint8_t>();

    auto run_seq_len = input_dims[0];
    PADDLE_ENFORCE_LE((size_t)run_seq_len, cudnn_rnn_cache->max_length_,
                      "cudnn running seq_len CAN not greater max_lengh");
    CUDNN_ENFORCE(platform::dynload::cudnnRNNBackwardData(
        handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
        cudnn_rnn_cache->y_desc_, out_data, cudnn_rnn_cache->dy_desc_,
        out_grad_data, cudnn_rnn_cache->dhy_desc_, last_h_grad_data,
        cudnn_rnn_cache->dcy_desc_, last_c_grad_data, cudnn_rnn_cache->w_desc_,
        weight_data, cudnn_rnn_cache->hx_desc_, init_h_data,
        cudnn_rnn_cache->cx_desc_, init_c_data, cudnn_rnn_cache->dx_desc_,
        in_grad_data, cudnn_rnn_cache->dhx_desc_, init_h_grad_data,
        cudnn_rnn_cache->dcx_desc_, init_c_grad_data, work_data,
        cudnn_rnn_cache->workspace_size_, reserve_data,
        cudnn_rnn_cache->reserve_size_));

    CUDNN_ENFORCE(platform::dynload::cudnnRNNBackwardWeights(
        handle, cudnn_rnn_cache->rnn_desc_, run_seq_len,
        cudnn_rnn_cache->x_desc_, input->data<T>(), cudnn_rnn_cache->hx_desc_,
        init_h->data<T>(), cudnn_rnn_cache->y_desc_, out->data<T>(),
        cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
        cudnn_rnn_cache->workspace_size_, cudnn_rnn_cache->dw_desc_,
        weight_grad->data<T>(), cudnn_rnn_cache->reserve_data_.data<uint8_t>(),
        cudnn_rnn_cache->reserve_size_));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cudnn_lstm, ops::CudnnLSTMGPUKernel<float>);
REGISTER_OP_CUDA_KERNEL(cudnn_lstm_grad, ops::CudnnLSTMGPUGradKernel<float>);
