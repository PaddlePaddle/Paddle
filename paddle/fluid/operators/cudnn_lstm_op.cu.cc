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
#include "paddle/fluid/platform/cudnn_helper.h"

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
    auto sequence_length = ctx.Attr<std::vector<int>>("sequence_length");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    int seq_length = x->dims()[0];
    int batch_size = x->dims()[1];
    int input_size = x->dims()[2];
    int weight_numel = w->numel();
    bool state_initialized = state_out->IsInitialized() ? true : false;

    size_t workspace_size;
    size_t reserve_size;

    platform::ScopedRNNBase rnn(seq_length, batch_size, input_size, hidden_size,
                                num_layers, dropout_prob, seed, weight_numel,
                                state_initialized, is_bidirec);
    rnn.Create<T>(handle, ctx.GetPlace(), sequence_length, &workspace_size,
                  &reserve_size, state_out);

    framework::Tensor workspace_data_;
    workspace_data_.Resize({static_cast<int64_t>(workspace_size)});
    workspace_data_.mutable_data<uint8_t>(ctx.GetPlace());

    auto *reserve_data = reserve->mutable_data<uint8_t>(
        {static_cast<int64_t>(reserve_size)}, ctx.GetPlace());

    if (is_test) {
      if (sequence_length.empty()) {
        // for inference
        // This interface is used when the input/output is unpadded.
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardInference(
            handle, rnn.rnn_desc(), seq_length, rnn.x_desc(), x_data,
            rnn.hx_desc(), init_h_data, rnn.cx_desc(), init_c_data,
            rnn.w_desc(), w_data, rnn.y_desc(), out_data, rnn.hy_desc(),
            last_h_data, rnn.cy_desc(), last_c_data,
            workspace_data_.data<uint8_t>(), workspace_size));
      } else {
#if CUDNN_VERSION >= 7201
        // for inference
        // This interface is used when the input/output is padded.
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnRNNForwardInferenceEx(
                handle, rnn.rnn_desc(), rnn.x_seq_desc(), x_data, rnn.hx_desc(),
                init_h_data, rnn.cx_desc(), init_c_data, rnn.w_desc(), w_data,
                rnn.y_seq_desc(), out_data, rnn.hy_desc(), last_h_data,
                rnn.cy_desc(), last_c_data, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr,
                workspace_data_.data<uint8_t>(), workspace_size));
#else
        PADDLE_ENFORCE_NOT_NULL(
            nullptr, platform::errors::Unavailable(
                         "The padded input is supported by "
                         "cudnnRNNForwardInferenceEx, but it only works when "
                         "the version of cudnn is larger than 7.2.1"));
#endif
      }
    } else {
      if (sequence_length.empty()) {
        // for train
        // This interface is used when the input/output is unpadded.
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardTraining(
            handle, rnn.rnn_desc(), seq_length, rnn.x_desc(), x_data,
            rnn.hx_desc(), init_h_data, rnn.cx_desc(), init_c_data,
            rnn.w_desc(), w_data, rnn.y_desc(), out_data, rnn.hy_desc(),
            last_h_data, rnn.cy_desc(), last_c_data,
            workspace_data_.data<uint8_t>(), workspace_size, reserve_data,
            reserve_size));
      } else {
#if CUDNN_VERSION >= 7201
        // for train
        // This interface is used when the input/output is padded.
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnRNNForwardTrainingEx(
                handle, rnn.rnn_desc(), rnn.x_seq_desc(), x_data, rnn.hx_desc(),
                init_h_data, rnn.cx_desc(), init_c_data, rnn.w_desc(), w_data,
                rnn.y_seq_desc(), out_data, rnn.hy_desc(), last_h_data,
                rnn.cy_desc(), last_c_data, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr,
                workspace_data_.data<uint8_t>(), workspace_size, reserve_data,
                reserve_size));
#else
        PADDLE_ENFORCE_NOT_NULL(
            nullptr, platform::errors::Unavailable(
                         "The padded input is supported by "
                         "cudnnRNNForwardTrainingEx, but it only works when "
                         "the version of cudnn is larger than 7.2.1"));
#endif
      }
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
    auto sequence_length = ctx.Attr<std::vector<int>>("sequence_length");

    int seq_length = input_dims[0];
    int batch_size = input->dims()[1];
    int input_size = input->dims()[2];
    int weight_numel = weight->numel();

    size_t workspace_size;
    size_t reserve_size;

    platform::ScopedRNNBase rnn(seq_length, batch_size, input_size, hidden_size,
                                num_layers, dropout_prob, seed, weight_numel,
                                true, is_bidirec);

    rnn.Create<T>(handle, ctx.GetPlace(), sequence_length, &workspace_size,
                  &reserve_size, const_cast<Tensor *>(state_out));

    framework::Tensor workspace_data_;
    workspace_data_.Resize({static_cast<int64_t>(workspace_size)});
    workspace_data_.mutable_data<uint8_t>(ctx.GetPlace());
    const uint8_t *reserve_data = reserve->data<uint8_t>();

    if (sequence_length.empty()) {
      // This interface is used when the input/output is unpadded.
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardData(
          handle, rnn.rnn_desc(), seq_length, rnn.y_desc(), out_data,
          rnn.y_desc(), out_grad_data, rnn.hy_desc(), last_h_grad_data,
          rnn.cy_desc(), last_c_grad_data, rnn.w_desc(), weight_data,
          rnn.hx_desc(), init_h_data, rnn.cx_desc(), init_c_data, rnn.x_desc(),
          in_grad_data, rnn.hx_desc(), init_h_grad_data, rnn.cx_desc(),
          init_c_grad_data, workspace_data_.data<uint8_t>(), workspace_size,
          const_cast<uint8_t *>(reserve_data), reserve_size));

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardWeights(
          handle, rnn.rnn_desc(), seq_length, rnn.x_desc(), input->data<T>(),
          rnn.hx_desc(), init_h->data<T>(), rnn.y_desc(), out->data<T>(),
          workspace_data_.data<uint8_t>(), workspace_size, rnn.w_desc(),
          weight_grad->data<T>(), const_cast<uint8_t *>(reserve_data),
          reserve_size));
    } else {
#if CUDNN_VERSION >= 7201
      // for train
      // This interface is used when the input/output is padded.
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardDataEx(
          handle, rnn.rnn_desc(), rnn.y_seq_desc(), out_data, rnn.y_seq_desc(),
          out_grad_data, nullptr, nullptr, rnn.hy_desc(), last_h_grad_data,
          rnn.cy_desc(), last_c_grad_data, rnn.w_desc(), weight_data,
          rnn.hx_desc(), init_h_data, rnn.cx_desc(), init_c_data,
          rnn.x_seq_desc(), in_grad_data, rnn.hx_desc(), init_h_grad_data,
          rnn.cx_desc(), init_c_grad_data, nullptr, nullptr,
          workspace_data_.data<uint8_t>(), workspace_size,
          const_cast<uint8_t *>(reserve_data), reserve_size));

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardWeightsEx(
          handle, rnn.rnn_desc(), rnn.x_seq_desc(), input->data<T>(),
          rnn.hx_desc(), init_h->data<T>(), rnn.y_seq_desc(), out->data<T>(),
          workspace_data_.data<uint8_t>(), workspace_size, rnn.w_desc(),
          weight_grad->data<T>(), const_cast<uint8_t *>(reserve_data),
          reserve_size));
#else
      PADDLE_ENFORCE_NOT_NULL(
          nullptr,
          platform::errors::Unavailable(
              "The padded input of rnn is supported by cudnnRNNBackwardDataEx, "
              "cudnnRNNBackwardWeightsEx, but it only works when the version "
              "of cudnn is larger than 7.2.1"));
#endif
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cudnn_lstm, ops::CudnnLSTMGPUKernel<float>,
                        ops::CudnnLSTMGPUKernel<double>);
REGISTER_OP_CUDA_KERNEL(cudnn_lstm_grad, ops::CudnnLSTMGPUGradKernel<float>,
                        ops::CudnnLSTMGPUGradKernel<double>);
