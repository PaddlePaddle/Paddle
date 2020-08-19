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
    bool time_major = ctx.Attr<bool>("time_major");
    auto is_seq_lengths = ctx.Attr<std::vector<int>>("sequence_length");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    int seq_len = x->dims()[0];
    int batch_size = x->dims()[1];
    int input_size = x->dims()[2];
    int weight_numel = w->numel();
    bool state_initialized = state_out->IsInitialized() ? true : false;
    int numDirections = is_bidirec ? 2 : 1;

    //  cudnnDataType_t cudnn_type = platform::ToCudnnDataType(
    //    framework::ToDataType(std::type_index(typeid(T))));
    cudnnDataType_t cudnn_type = platform::CudnnDataType<T>::type;
    // ------------------- cudnn x, y descriptors ---------------------
    cudnnTensorDescriptor_t *x_desc_ = new cudnnTensorDescriptor_t[seq_len];
    cudnnTensorDescriptor_t *y_desc_ = new cudnnTensorDescriptor_t[seq_len];

    std::vector<int> dims_x = {batch_size, input_size, 1};
    std::vector<int> strides_x = {input_size, 1, 1};

    std::vector<int> dims_y = {batch_size, hidden_size * numDirections, 1};
    std::vector<int> strides_y = {hidden_size * numDirections, 1, 1};

      platform::ScopedTensorDescriptor x_desc;
      platform::ScopedTensorDescriptor y_desc;
    for (int i = 0; i < seq_len; ++i) {
      x_desc_[i] = x_desc.descriptor<T>(dims_x, strides_x);
      y_desc_[i] = y_desc.descriptor<T>(dims_y, strides_y);
    }

    platform::ScopedRNNSeqTensorDescriptor x_seq_desc;
    cudnnRNNDataDescriptor_t x_seq_desc_ = x_seq_desc.descriptor<T>(
        input_size, batch_size, input_size, time_major, is_seq_lengths);
    platform::ScopedRNNSeqTensorDescriptor y_seq_desc;
    cudnnRNNDataDescriptor_t y_seq_desc_ = y_seq_desc.descriptor<T>(
        hidden_size * numDirections, batch_size, hidden_size * numDirections,
        time_major, is_seq_lengths);

    // ------------------- cudnn hx, hy, cx, cy descriptors----------
    std::vector<int> dims_hx = {num_layers * numDirections, batch_size,
                                hidden_size};
    std::vector<int> strides_hx = {hidden_size * batch_size, hidden_size, 1};

    platform::ScopedTensorDescriptor hx_desc;
    cudnnTensorDescriptor_t hx_desc_ =
        hx_desc.descriptor<T>(dims_hx, strides_hx);

    platform::ScopedTensorDescriptor cx_desc;
    cudnnTensorDescriptor_t cx_desc_ =
        cx_desc.descriptor<T>(dims_hx, strides_hx);

    platform::ScopedTensorDescriptor hy_desc;
    cudnnTensorDescriptor_t hy_desc_ =
        hy_desc.descriptor<T>(dims_hx, strides_hx);

    platform::ScopedTensorDescriptor cy_desc;
    cudnnTensorDescriptor_t cy_desc_ =
        cy_desc.descriptor<T>(dims_hx, strides_hx);

    // ------------------- cudnn dropout descriptors ---------------------
    platform::ScopedDropoutDescriptor dropout_desc;
    cudnnDropoutDescriptor_t dropout_desc_ =
        dropout_desc.descriptor(handle, ctx.GetPlace(), state_initialized,
                                dropout_prob, state_out, seed);

    // ------------------- cudnn rnn descriptors ---------------------
    platform::ScopedRNNDescriptor rnn_desc;
    cudnnRNNDescriptor_t rnn_desc_ = rnn_desc.descriptor();

#if CUDNN_VERSION >= 6000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor_v6(
        handle, rnn_desc_, hidden_size, num_layers, dropout_desc_,
        CUDNN_LINEAR_INPUT,
        is_bidirec ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        CUDNN_RNN_ALGO_STANDARD, cudnn_type));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor(
        rnn_desc_, hidden_size, num_layers, dropout_desc_, CUDNN_LINEAR_INPUT,
        is_bidirec ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        cudnn_type));
#endif

    // ------------------- cudnn weights_size ---------------------
    size_t weights_size_;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc_, x_desc_[0], &weights_size_, cudnn_type));

    PADDLE_ENFORCE_EQ(
        weights_size_, sizeof(T) * weight_numel,
        platform::errors::InvalidArgument(
            "The cudnn lstm and setting weight size should be same."));

    // ------------------- cudnn weight descriptors ---------------------
    platform::ScopedFilterDescriptor w_desc;
    platform::DataLayout layout = platform::DataLayout::kNCHW;
    int dim_tmp = weights_size_ / sizeof(T);
    std::vector<int> dim_w = {dim_tmp, 1, 1};
    cudnnFilterDescriptor_t w_desc_ = w_desc.descriptor<T>(layout, dim_w);

    // ------------------- cudnn workspace, reserve size ---------------------
    size_t workspace_size_;
    size_t reserve_size;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNWorkspaceSize(
        handle, rnn_desc_, seq_len, x_desc_, &workspace_size_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetRNNTrainingReserveSize(
            handle, rnn_desc_, seq_len, x_desc_, &reserve_size));

    // LOG(INFO) << "reserve_size" << reserve_size;

    framework::Tensor workspace_data_;
    workspace_data_.Resize({static_cast<int64_t>(workspace_size_)});
    workspace_data_.mutable_data<uint8_t>(ctx.GetPlace());

    auto *reserve_data = reserve->mutable_data<uint8_t>(
        {static_cast<int64_t>(reserve_size)}, ctx.GetPlace());

    if (is_test) {
      // for inference
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardInferenceEx(
          handle, rnn_desc_, x_seq_desc_, x_data, hx_desc_, init_h_data,
          cx_desc_, init_c_data, w_desc_, w_data, x_seq_desc_, out_data,
          hy_desc_, last_h_data, cy_desc_, last_c_data, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          workspace_data_.data<uint8_t>(), workspace_size_));
    } else {
    //   for train
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardTrainingEx(
          handle, rnn_desc_, x_seq_desc_, x_data, hx_desc_, init_h_data,
          cx_desc_, init_c_data, w_desc_, w_data, y_seq_desc_, out_data,
          hy_desc_, last_h_data, cy_desc_, last_c_data, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          workspace_data_.data<uint8_t>(), workspace_size_,
          const_cast<uint8_t *>(reserve_data), reserve_size));
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
    bool time_major = ctx.Attr<bool>("time_major");
    auto is_seq_lengths = ctx.Attr<std::vector<int>>("sequence_length");

    int seq_len = input_dims[0];
    int batch_size = input->dims()[1];
    int input_size = input->dims()[2];
    int weight_numel = weight->numel();
    int numDirections = is_bidirec ? 2 : 1;

    //    cudnnDataType_t cudnn_type = platform::ToCudnnDataType(
    //      framework::ToDataType(std::type_index(typeid(T))));
    cudnnDataType_t cudnn_type = platform::CudnnDataType<T>::type;

    // ------------------- cudnn x, y descriptors ---------------------
    cudnnTensorDescriptor_t *x_desc_ = new cudnnTensorDescriptor_t[seq_len];
    cudnnTensorDescriptor_t *y_desc_ = new cudnnTensorDescriptor_t[seq_len];

    std::vector<int> dims_x = {batch_size, input_size, 1};
    std::vector<int> strides_x = {input_size, 1, 1};

    std::vector<int> dims_y = {batch_size, hidden_size * numDirections, 1};
    std::vector<int> strides_y = {hidden_size * numDirections, 1, 1};

      platform::ScopedTensorDescriptor x_desc;
      platform::ScopedTensorDescriptor y_desc;
    for (int i = 0; i < seq_len; ++i) {
      x_desc_[i] = x_desc.descriptor<T>(dims_x, strides_x);
      y_desc_[i] = y_desc.descriptor<T>(dims_y, strides_y);
    }

    platform::ScopedRNNSeqTensorDescriptor x_seq_desc;
    cudnnRNNDataDescriptor_t x_seq_desc_ = x_seq_desc.descriptor<T>(
        input_size, batch_size, input_size, time_major, is_seq_lengths);
    platform::ScopedRNNSeqTensorDescriptor y_seq_desc;
    cudnnRNNDataDescriptor_t y_seq_desc_ = y_seq_desc.descriptor<T>(
        hidden_size * numDirections, batch_size, hidden_size * numDirections,
        time_major, is_seq_lengths);

    // ------------------- cudnn hx, hy, cx, cy descriptors----------
    std::vector<int> dims_hx = {num_layers * numDirections, batch_size,
                                hidden_size};
    std::vector<int> strides_hx = {hidden_size * batch_size, hidden_size, 1};

    platform::ScopedTensorDescriptor hx_desc;
    cudnnTensorDescriptor_t hx_desc_ =
        hx_desc.descriptor<T>(dims_hx, strides_hx);

    platform::ScopedTensorDescriptor cx_desc;
    cudnnTensorDescriptor_t cx_desc_ =
        cx_desc.descriptor<T>(dims_hx, strides_hx);

    platform::ScopedTensorDescriptor hy_desc;
    cudnnTensorDescriptor_t hy_desc_ =
        hy_desc.descriptor<T>(dims_hx, strides_hx);

    platform::ScopedTensorDescriptor cy_desc;
    cudnnTensorDescriptor_t cy_desc_ =
        cy_desc.descriptor<T>(dims_hx, strides_hx);

    // ------------------- cudnn dropout descriptors ---------------------
    platform::ScopedDropoutDescriptor dropout_desc;
    cudnnDropoutDescriptor_t dropout_desc_ =
        dropout_desc.descriptor(handle, ctx.GetPlace(), true, dropout_prob,
                                const_cast<Tensor *>(state_out), seed);

    // ------------------- cudnn rnn descriptors ---------------------
    platform::ScopedRNNDescriptor rnn_desc;
    cudnnRNNDescriptor_t rnn_desc_ = rnn_desc.descriptor();

#if CUDNN_VERSION >= 6000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor_v6(
        handle, rnn_desc_, hidden_size, num_layers, dropout_desc_,
        CUDNN_LINEAR_INPUT,
        is_bidirec ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        CUDNN_RNN_ALGO_STANDARD, cudnn_type));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor(
        rnn_desc_, hidden_size, num_layers, dropout_desc_, CUDNN_LINEAR_INPUT,
        is_bidirec ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        cudnn_type));
#endif

    // ------------------- cudnn weights_size ---------------------
    size_t weights_size_;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc_, x_desc_[0], &weights_size_, cudnn_type));

    PADDLE_ENFORCE_EQ(
        weights_size_, sizeof(T) * weight_numel,
        platform::errors::InvalidArgument(
            "The cudnn lstm and setting weight size should be same."));

    // ------------------- cudnn weight descriptors ---------------------
    platform::ScopedFilterDescriptor w_desc;
    platform::DataLayout layout = platform::DataLayout::kNCHW;
    int dim_tmp = weights_size_ / sizeof(T);
    std::vector<int> dim_w = {dim_tmp, 1, 1};
    cudnnFilterDescriptor_t w_desc_ = w_desc.descriptor<T>(layout, dim_w);

    // ------------------- cudnn workspace, reserve size ---------------------
    size_t workspace_size_;
    size_t reserve_size;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNWorkspaceSize(
        handle, rnn_desc_, seq_len, x_desc_, &workspace_size_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetRNNTrainingReserveSize(
            handle, rnn_desc_, seq_len, x_desc_, &reserve_size));

    LOG(INFO) << "reserve_size" << reserve_size;

    framework::Tensor workspace_data_;
    workspace_data_.Resize({static_cast<int64_t>(workspace_size_)});
    workspace_data_.mutable_data<uint8_t>(ctx.GetPlace());
    const uint8_t *reserve_data = reserve->data<uint8_t>();

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardDataEx(
        handle, rnn_desc_, y_seq_desc_, out_data, y_seq_desc_, out_grad_data,
        nullptr, nullptr, hy_desc_, last_h_grad_data, cy_desc_,
        last_c_grad_data, w_desc_, weight_data, hx_desc_, init_h_data, cx_desc_,
        init_c_data, x_seq_desc_, in_grad_data, hx_desc_, init_h_grad_data,
        cx_desc_, nullptr, nullptr, init_c_grad_data,
        workspace_data_.data<uint8_t>(), workspace_size_,
        const_cast<uint8_t *>(reserve_data), reserve_size));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardWeightsEx(
        handle, rnn_desc_, x_seq_desc_, input->data<T>(), hx_desc_,
        init_h->data<T>(), y_seq_desc_, out->data<T>(),
        workspace_data_.data<uint8_t>(), workspace_size_, w_desc_,
        weight_grad->data<T>(), const_cast<uint8_t *>(reserve_data),
        reserve_size));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cudnn_lstm, ops::CudnnLSTMGPUKernel<float>,
                        ops::CudnnLSTMGPUKernel<double>);
REGISTER_OP_CUDA_KERNEL(cudnn_lstm_grad, ops::CudnnLSTMGPUGradKernel<float>,
                        ops::CudnnLSTMGPUGradKernel<double>);
