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
#include "paddle/fluid/platform/cudnn_desc.h"

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
    Tensor *last_h = ctx.Output<Tensor>("LastH");
    Tensor *last_c = ctx.Output<Tensor>("LastC");
    Tensor *reserve = ctx.Output<Tensor>("Reserve");
    Tensor *state_out = ctx.Output<Tensor>("StateOut");

    const T *x_data = x->data<T>();
    const T *init_h_data = init_h->data<T>();
    const T *init_c_data = init_c->data<T>();

    const T *w_data = w->data<T>();

    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    T *last_h_data = last_h->mutable_data<T>(ctx.GetPlace());
    T *last_c_data = last_c->mutable_data<T>(ctx.GetPlace());

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    bool is_test = ctx.Attr<bool>("is_test");
    int seed = ctx.Attr<int>("seed");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    CudnnRNNCache *cudnn_rnn_cache = new CudnnRNNCache();

    auto input_w_numel = w->numel();
    auto seq_len = x->dims()[0];
    auto batch_size = x->dims()[1];
    auto input_dim = x->dims()[2];
    size_t reserve_size;
    bool state_initialized = state_out->IsInitialized() ? true : false;
    cudnnDataType_t cudnn_type = platform::ToCudnnDataType(
        framework::ToDataType(std::type_index(typeid(T))));
    cudnn_rnn_cache->init(handle, ctx.GetPlace(), seq_len, batch_size,
                          input_dim, hidden_size, num_layers, dropout_prob,
                          is_bidirec, seed, input_w_numel, &reserve_size,
                          state_out, state_initialized, cudnn_type);

    auto *reserve_data = reserve->mutable_data<uint8_t>(
        {static_cast<int64_t>(reserve_size)}, ctx.GetPlace());

    if (is_test) {
      // for inference
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardInference(
          handle, cudnn_rnn_cache->rnn_desc_, seq_len, cudnn_rnn_cache->x_desc_,
          x_data, cudnn_rnn_cache->hx_desc_, init_h_data,
          cudnn_rnn_cache->cx_desc_, init_c_data, cudnn_rnn_cache->w_desc_,
          w_data, cudnn_rnn_cache->y_desc_, out_data, cudnn_rnn_cache->hy_desc_,
          last_h_data, cudnn_rnn_cache->cy_desc_, last_c_data,
          cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
          cudnn_rnn_cache->workspace_size_));
    } else {
      // for train
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardTraining(
          handle, cudnn_rnn_cache->rnn_desc_, seq_len, cudnn_rnn_cache->x_desc_,
          x_data, cudnn_rnn_cache->hx_desc_, init_h_data,
          cudnn_rnn_cache->cx_desc_, init_c_data, cudnn_rnn_cache->w_desc_,
          w_data, cudnn_rnn_cache->y_desc_, out_data, cudnn_rnn_cache->hy_desc_,
          last_h_data, cudnn_rnn_cache->cy_desc_, last_c_data,
          cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
          cudnn_rnn_cache->workspace_size_, reserve_data, reserve_size));
    }
    delete cudnn_rnn_cache;
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
    auto *reserve = ctx.Input<Tensor>("Reserve");
    auto *state_out = ctx.Input<Tensor>("StateOut");

    auto *out = ctx.Input<Tensor>("Out");
    auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *last_h_grad = ctx.Input<Tensor>(framework::GradVarName("LastH"));
    auto *last_c_grad = ctx.Input<Tensor>(framework::GradVarName("LastC"));

    auto *in_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto *weight_grad = ctx.Output<Tensor>(framework::GradVarName("W"));
    auto *init_h_grad = ctx.Output<Tensor>(framework::GradVarName("InitH"));
    auto *init_c_grad = ctx.Output<Tensor>(framework::GradVarName("InitC"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    auto input_dims = input->dims();
    auto init_h_dims = init_h->dims();
    auto init_c_dims = init_c->dims();

    auto *weight_data = weight->data<T>();
    auto *init_h_data = init_h->data<T>();
    auto *init_c_data = init_c->data<T>();
    auto *out_data = out->data<T>();
    auto *out_grad_data = out_grad->data<T>();
    auto *last_h_grad_data = last_h_grad->data<T>();
    auto *last_c_grad_data = last_c_grad->data<T>();

    math::SetConstant<paddle::platform::CUDADeviceContext, T> zero;
    weight_grad->mutable_data<T>(ctx.GetPlace());
    zero(dev_ctx, weight_grad, static_cast<T>(0.0));

    in_grad->mutable_data<T>(input_dims, ctx.GetPlace());
    auto *in_grad_data = in_grad->data<T>();

    init_h_grad->mutable_data<T>(init_h_dims, ctx.GetPlace());
    auto *init_h_grad_data = init_h_grad->data<T>();

    init_c_grad->mutable_data<T>(init_c_dims, ctx.GetPlace());
    auto *init_c_grad_data = init_c_grad->data<T>();

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    int seed = ctx.Attr<int>("seed");

    CudnnRNNCache *cudnn_rnn_cache = new CudnnRNNCache();

    auto input_w_numel = weight->numel();
    auto seq_len = input_dims[0];
    auto batch_size = input->dims()[1];
    auto input_dim = input->dims()[2];
    size_t reserve_size;
    cudnnDataType_t cudnn_type = platform::ToCudnnDataType(
        framework::ToDataType(std::type_index(typeid(T))));
    cudnn_rnn_cache->init(handle, ctx.GetPlace(), seq_len, batch_size,
                          input_dim, hidden_size, num_layers, dropout_prob,
                          is_bidirec, seed, input_w_numel, &reserve_size,
                          const_cast<Tensor *>(state_out), true, cudnn_type);

    auto work_data = cudnn_rnn_cache->workspace_data_.data<uint8_t>();
    const uint8_t *reserve_data = reserve->data<uint8_t>();

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardData(
        handle, cudnn_rnn_cache->rnn_desc_, seq_len, cudnn_rnn_cache->y_desc_,
        out_data, cudnn_rnn_cache->y_desc_, out_grad_data,
        cudnn_rnn_cache->hy_desc_, last_h_grad_data, cudnn_rnn_cache->cy_desc_,
        last_c_grad_data, cudnn_rnn_cache->w_desc_, weight_data,
        cudnn_rnn_cache->hx_desc_, init_h_data, cudnn_rnn_cache->cx_desc_,
        init_c_data, cudnn_rnn_cache->x_desc_, in_grad_data,
        cudnn_rnn_cache->hx_desc_, init_h_grad_data, cudnn_rnn_cache->cx_desc_,
        init_c_grad_data, work_data, cudnn_rnn_cache->workspace_size_,
        const_cast<uint8_t *>(reserve_data), reserve_size));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardWeights(
        handle, cudnn_rnn_cache->rnn_desc_, seq_len, cudnn_rnn_cache->x_desc_,
        input->data<T>(), cudnn_rnn_cache->hx_desc_, init_h->data<T>(),
        cudnn_rnn_cache->y_desc_, out->data<T>(),
        cudnn_rnn_cache->workspace_data_.data<uint8_t>(),
        cudnn_rnn_cache->workspace_size_, cudnn_rnn_cache->w_desc_,
        weight_grad->data<T>(), const_cast<uint8_t *>(reserve_data),
        reserve_size));
    delete cudnn_rnn_cache;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cudnn_lstm, ops::CudnnLSTMGPUKernel<float>,
                        ops::CudnnLSTMGPUKernel<double>);
REGISTER_OP_CUDA_KERNEL(cudnn_lstm_grad, ops::CudnnLSTMGPUGradKernel<float>,
                        ops::CudnnLSTMGPUGradKernel<double>);
