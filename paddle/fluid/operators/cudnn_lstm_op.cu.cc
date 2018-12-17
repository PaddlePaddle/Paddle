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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

struct CudnnRNNCache {
  CudnnRNNCache() {
    x_desc_ = NULL;
    y_desc_ = NULL;
    dx_desc_ = NULL;
    dy_desc_ = NULL;
  }
  ~CudnnRNNCache() { release(); }

  cudnnRNNDescriptor_t rnn_desc_;
  cudnnTensorDescriptor_t *x_desc_;
  cudnnTensorDescriptor_t *y_desc_;
  cudnnTensorDescriptor_t *dx_desc_;
  cudnnTensorDescriptor_t *dy_desc_;

  cudnnTensorDescriptor_t hx_desc_;
  cudnnTensorDescriptor_t cx_desc_;
  cudnnTensorDescriptor_t hy_desc_;
  cudnnTensorDescriptor_t cy_desc_;

  cudnnTensorDescriptor_t dhx_desc_;
  cudnnTensorDescriptor_t dcx_desc_;
  cudnnTensorDescriptor_t dhy_desc_;
  cudnnTensorDescriptor_t dcy_desc_;

  cudnnTensorDescriptor_t output_x_desc_;
  cudnnTensorDescriptor_t output_y_desc_;

  cudnnDropoutDescriptor_t dropout_desc_;

  size_t weights_size_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnFilterDescriptor_t dw_desc_;

  size_t workspace_size_;
  size_t reserve_size_;
  Tensor reserve_data_;
  Tensor workspace_data_;

  Tensor dropout_state_;

  size_t max_length_;

  float dropout_prob_;
  bool is_bidirec_;

  int batch_size_;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  int seed_;

  void init(cudnnHandle_t handle, const framework::ExecutionContext &ctx,
            size_t max_len, int batch_size, int input_size, int hidden_size,
            int num_layers, float dropout_prob, bool is_bidirec, int seed,
            int weight_numel) {
    max_length_ = max_len;
    batch_size_ = batch_size;
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    num_layers_ = num_layers;
    dropout_prob_ = dropout_prob;
    is_bidirec_ = is_bidirec;
    seed_ = seed;

    x_desc_ = new cudnnTensorDescriptor_t[max_length_];
    y_desc_ = new cudnnTensorDescriptor_t[max_length_];
    dx_desc_ = new cudnnTensorDescriptor_t[max_length_];
    dy_desc_ = new cudnnTensorDescriptor_t[max_length_];
    int dim_a[3];
    int stride_a[3];

    for (size_t i = 0; i < max_length_; ++i) {
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&x_desc_[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&y_desc_[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&dx_desc_[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateTensorDescriptor(&dy_desc_[i]));
      dim_a[0] = batch_size_;
      dim_a[1] = input_size_;
      dim_a[2] = 1;

      stride_a[0] = dim_a[2] * dim_a[1];
      stride_a[1] = dim_a[2];
      stride_a[2] = 1;
      CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
          x_desc_[i], CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
      CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
          dx_desc_[i], CUDNN_DATA_FLOAT, 3, dim_a, stride_a));

      dim_a[0] = batch_size_;
      dim_a[1] = is_bidirec_ ? hidden_size_ * 2 : hidden_size_;
      dim_a[2] = 1;

      stride_a[0] = dim_a[2] * dim_a[1];
      stride_a[1] = dim_a[2];
      stride_a[2] = 1;

      CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
          y_desc_[i], CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
      CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
          dy_desc_[i], CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    }

    dim_a[0] = num_layers_ * (is_bidirec_ ? 2 : 1);
    dim_a[1] = batch_size_;
    dim_a[2] = hidden_size_;

    stride_a[0] = dim_a[2] * dim_a[1];
    stride_a[1] = dim_a[2];
    stride_a[2] = 1;

    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&hx_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&cx_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&hy_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&cy_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&dhx_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&dcx_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&dhy_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&dcy_desc_));

    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        hx_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        cx_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        hy_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        cy_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        dhx_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        dcx_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        dhy_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));
    CUDNN_ENFORCE(platform::dynload::cudnnSetTensorNdDescriptor(
        dcy_desc_, CUDNN_DATA_FLOAT, 3, dim_a, stride_a));

    CUDNN_ENFORCE(
        platform::dynload::cudnnCreateDropoutDescriptor(&dropout_desc_));

    size_t state_size;
    CUDNN_ENFORCE(
        platform::dynload::cudnnDropoutGetStatesSize(handle, &state_size);
        dropout_state_.Resize({static_cast<int64_t>(state_size)}));
    auto *dropout_state_data =
        dropout_state_.mutable_data<uint8_t>(ctx.GetPlace());
    CUDNN_ENFORCE(platform::dynload::cudnnSetDropoutDescriptor(
        dropout_desc_, handle, dropout_prob_, dropout_state_data, state_size,
        seed_));

    CUDNN_ENFORCE(platform::dynload::cudnnCreateRNNDescriptor(&rnn_desc_));

#if CUDNN_VERSION >= 6000
    CUDNN_ENFORCE(platform::dynload::cudnnSetRNNDescriptor_v6(
        handle, rnn_desc_, hidden_size_, num_layers_, dropout_desc_,
        CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
#else
    CUDNN_ENFORCE(platform::dynload::cudnnSetRNNDescriptor(
        rnn_desc_, hidden_size_, num_layers_, dropout_desc_, CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LSTM,
        CUDNN_DATA_FLOAT));
#endif

    CUDNN_ENFORCE(platform::dynload::cudnnCreateFilterDescriptor(&w_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateFilterDescriptor(&dw_desc_));

    CUDNN_ENFORCE(platform::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc_, x_desc_[0], &weights_size_, CUDNN_DATA_FLOAT));

    PADDLE_ENFORCE_EQ(weights_size_, sizeof(float) * weight_numel,
                      "cudnn lstm weight size should be SAME");
    int dim_w[3];
    dim_w[0] = weights_size_ / sizeof(float);
    dim_w[1] = 1;
    dim_w[2] = 1;
    CUDNN_ENFORCE(platform::dynload::cudnnSetFilterNdDescriptor(
        w_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dim_w));
    CUDNN_ENFORCE(platform::dynload::cudnnSetFilterNdDescriptor(
        dw_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dim_w));

    CUDNN_ENFORCE(platform::dynload::cudnnGetRNNWorkspaceSize(
        handle, rnn_desc_, max_length_, x_desc_, &workspace_size_));
    CUDNN_ENFORCE(platform::dynload::cudnnGetRNNTrainingReserveSize(
        handle, rnn_desc_, max_length_, x_desc_, &reserve_size_));

    reserve_data_.Resize({static_cast<int64_t>(reserve_size_)});
    reserve_data_.mutable_data<uint8_t>(ctx.GetPlace());

    workspace_data_.Resize({static_cast<int64_t>(workspace_size_)});
    workspace_data_.mutable_data<uint8_t>(ctx.GetPlace());
  }

  void release() {
    for (size_t i = 0; i < max_length_; ++i) {
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(x_desc_[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(y_desc_[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(dx_desc_[i]));
      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyTensorDescriptor(dy_desc_[i]));
    }

    delete[] x_desc_;
    delete[] y_desc_;
    delete[] dx_desc_;
    delete[] dy_desc_;

    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(hx_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(cx_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(hy_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(cy_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(dhx_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(dcx_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(dhy_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(dcy_desc_));

    CUDNN_ENFORCE(
        platform::dynload::cudnnDestroyDropoutDescriptor(dropout_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyRNNDescriptor(rnn_desc_));

    CUDNN_ENFORCE(platform::dynload::cudnnDestroyFilterDescriptor(w_desc_));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyFilterDescriptor(dw_desc_));
  }
};

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
      cudnn_rnn_cache->init(handle, ctx, max_len, batch_size, input_size,
                            hidden_size, num_layers, dropout_prob, is_bidirec,
                            seed, input_w_numel);
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
    auto weight_dims = weight->dims();
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
