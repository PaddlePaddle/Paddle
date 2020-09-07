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
bool is_continuous(const std::vector<const Tensor *> &weight_list) {
  bool continuous = true;
  for (size_t i = 0; i < weight_list.size() - 1; ++i) {
    const T *in_data = weight_list[i]->data<T>();
    const T *in_after_data = weight_list[i + 1]->data<T>();
    auto in_size = weight_list[i]->numel();
    bool temp = in_data + in_size * sizeof(T) == in_after_data;
    // PADDLE_ENFORCE_EQ(in_data + in_size * sizeof(T), in_after_data,
    //                  platform::errors::Unimplemented(
    //                      "The memory space of weights needs to be continuous.
    //                      "
    //                      "The op can be called."));
    continuous = continuous && temp;
  }
  return continuous;
}

int size_sum(const std::vector<const Tensor *> &weight_list) {
  int size = 0;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    auto in_size = weight_list[i]->numel();
    size += in_size;
  }
  return size;
}

template <typename T>
void copy_weight_forward(const platform::Place &place, cudaStream_t stream,
                         const std::vector<const Tensor *> &weight_list,
                         Tensor *weight) {
  LOG(INFO) << "weight_data";

  auto weight_data = weight->data<T>();
  int weight_offset = 0;
  LOG(INFO) << "size: " << weight_list.size();
  for (size_t i = 0; i < weight_list.size(); ++i) {
    const T *in_data = weight_list[i]->data<T>();
    auto in_size = weight_list[i]->numel();
    LOG(INFO) << "weight_size: " << in_size
              << " weight_offset: " << weight_offset;

    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, weight->place()),
                 weight_data + weight_offset,
                 BOOST_GET_CONST(platform::CUDAPlace, weight_list[i]->place()),
                 in_data, in_size * sizeof(T), stream);
    weight_offset += in_size;
    LOG(INFO) << "weight_offset: " << weight_offset;
  }
}

template <typename T>
void copy_weight_grad(const platform::Place &place, cudaStream_t stream,
                      std::vector<Tensor *> *weight_grad,
                      const std::vector<const Tensor *> &weight_input,
                      const Tensor *weight) {
  int weight_offset = 0;
  LOG(INFO) << "size: " << weight_input.size();
  auto *weight_data = weight->data<T>();
  for (size_t i = 0; i < weight_input.size(); ++i) {
    auto in_size = weight_input[i]->numel();
    LOG(INFO) << "weight_size: " << in_size;
    T *weight_grad_data = (*weight_grad)[i]->mutable_data<T>(
        framework::make_ddim({in_size}), place);
    LOG(INFO) << "weight_offset: " << weight_offset;
    const T *src = weight_data + weight_offset;

    memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, (*weight_grad)[i]->place()),
        weight_grad_data, BOOST_GET_CONST(platform::CUDAPlace, weight->place()),
        src, in_size * sizeof(T), stream);
    weight_offset += in_size;
    LOG(INFO) << "weight_offset: " << weight_offset;
  }
}

template <typename T>
class CudnnLSTMGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *x = ctx.Input<Tensor>("Input");
    const Tensor *init_h = ctx.Input<Tensor>("InitH");
    const Tensor *init_c = ctx.Input<Tensor>("InitC");

    Tensor *out = ctx.Output<Tensor>("Out");
    Tensor *last_h = ctx.Output<Tensor>("LastH");
    Tensor *last_c = ctx.Output<Tensor>("LastC");
    Tensor *reserve = ctx.Output<Tensor>("Reserve");
    Tensor *state_out = ctx.Output<Tensor>("StateOut");

    const T *x_data = x->data<T>();
    const T *init_h_data = init_h->data<T>();
    const T *init_c_data = init_c->data<T>();

    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    T *last_h_data = last_h->mutable_data<T>(ctx.GetPlace());
    T *last_c_data = last_c->mutable_data<T>(ctx.GetPlace());

    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    bool continuous = is_continuous<T>(weight_list);
    int weight_numel = size_sum(weight_list);
    LOG(INFO) << "continuous: " << continuous;
    LOG(INFO) << " weight_numel: " << weight_numel;

    auto place = ctx.GetPlace();
    LOG(INFO) << "place" << platform::is_gpu_place(place);
    auto stream = reinterpret_cast<const platform::CUDADeviceContext &>(
                      ctx.device_context())
                      .stream();
    Tensor weight_whole;
    weight_whole.mutable_data<T>({weight_numel}, place);
    T *w_data = nullptr;

    if (!continuous) {
      LOG(INFO) << "copy_weight_forward";
      copy_weight_forward<T>(place, stream, weight_list, &weight_whole);
      LOG(INFO) << "w_data";
      w_data = weight_whole.data<T>();
    } else {
      w_data = const_cast<T *>(weight_list[0]->data<T>());
    }

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
    bool state_initialized = state_out->IsInitialized() ? true : false;

    size_t workspace_size;
    size_t reserve_size;

    LOG(INFO) << "rnn initialize";
    platform::ScopedRNNBase rnn(seq_length, batch_size, input_size, hidden_size,
                                num_layers, dropout_prob, seed, weight_numel,
                                state_initialized, is_bidirec);
    LOG(INFO) << "rnn creat";
    rnn.Create<T>(handle, ctx.GetPlace(), sequence_length, &workspace_size,
                  &reserve_size, state_out);

    LOG(INFO) << "rnn create end";
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
        LOG(INFO) << "lstm forward compute";
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
    LOG(INFO) << "lstm forward";
  }
};

template <typename T>
class CudnnLSTMGPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input = ctx.Input<Tensor>("Input");
    auto *init_h = ctx.Input<Tensor>("InitH");
    auto *init_c = ctx.Input<Tensor>("InitC");
    auto *reserve = ctx.Input<Tensor>("Reserve");
    auto *state_out = ctx.Input<Tensor>("StateOut");

    auto *out = ctx.Input<Tensor>("Out");
    auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *last_h_grad = ctx.Input<Tensor>(framework::GradVarName("LastH"));
    auto *last_c_grad = ctx.Input<Tensor>(framework::GradVarName("LastC"));

    auto *in_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto *init_h_grad = ctx.Output<Tensor>(framework::GradVarName("InitH"));
    auto *init_c_grad = ctx.Output<Tensor>(framework::GradVarName("InitC"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto place = ctx.GetPlace();

    auto input_dims = input->dims();
    auto init_h_dims = init_h->dims();
    auto init_c_dims = init_c->dims();

    auto *init_h_data = init_h->data<T>();
    auto *init_c_data = init_c->data<T>();
    auto *out_data = out->data<T>();
    auto *out_grad_data = out_grad->data<T>();
    auto *last_h_grad_data = last_h_grad->data<T>();
    auto *last_c_grad_data = last_c_grad->data<T>();

    LOG(INFO) << "lstm grad";
    Tensor weight_grad;
    math::SetConstant<paddle::platform::CUDADeviceContext, T> zero;
    weight_grad.mutable_data<T>(ctx.GetPlace());
    zero(dev_ctx, &weight_grad, static_cast<T>(0.0));
    LOG(INFO) << "grad zero";

    in_grad->mutable_data<T>(input_dims, ctx.GetPlace());
    auto *in_grad_data = in_grad->data<T>();

    init_h_grad->mutable_data<T>(init_h_dims, ctx.GetPlace());
    auto *init_h_grad_data = init_h_grad->data<T>();

    init_c_grad->mutable_data<T>(init_c_dims, ctx.GetPlace());
    auto *init_c_grad_data = init_c_grad->data<T>();

    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    int weight_numel = size_sum(weight_list);
    T *weight_data = const_cast<T *>(weight_list[0]->data<T>());

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    int seed = ctx.Attr<int>("seed");
    auto sequence_length = ctx.Attr<std::vector<int>>("sequence_length");

    int seq_length = input_dims[0];
    int batch_size = input->dims()[1];
    int input_size = input->dims()[2];

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
          weight_grad.data<T>(), const_cast<uint8_t *>(reserve_data),
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
          weight_grad.data<T>(), const_cast<uint8_t *>(reserve_data),
          reserve_size));
      LOG(INFO) << "lstm grad compute";
#else
      PADDLE_ENFORCE_NOT_NULL(
          nullptr,
          platform::errors::Unavailable(
              "The padded input of rnn is supported by cudnnRNNBackwardDataEx, "
              "cudnnRNNBackwardWeightsEx, but it only works when the version "
              "of cudnn is larger than 7.2.1"));
#endif
    }
    auto stream = reinterpret_cast<const platform::CUDADeviceContext &>(
                      ctx.device_context())
                      .stream();
    LOG(INFO) << "copy_weight_grad";
    auto weight_grad_list = ctx.MultiOutput<framework::Tensor>(
        framework::GradVarName("WeightList"));
    copy_weight_grad<T>(place, stream, &weight_grad_list, weight_list,
                        &weight_grad);
    LOG(INFO) << "w_data_grad";
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cudnn_lstm, ops::CudnnLSTMGPUKernel<float>,
                        ops::CudnnLSTMGPUKernel<double>);
REGISTER_OP_CUDA_KERNEL(cudnn_lstm_grad, ops::CudnnLSTMGPUGradKernel<float>,
                        ops::CudnnLSTMGPUGradKernel<double>);
