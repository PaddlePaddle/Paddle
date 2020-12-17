/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/dynload/cudnn.h"

namespace paddle {
namespace platform {
class CUDADeviceContext;
struct CUDAPlace;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

class RNNDescriptors {
 public:
  RNNDescriptors(int seq_length, int batch_size, int input_size,
                 int hidden_size, int num_layers, float dropout_prob, int seed,
                 int weight_numel, cudnnRNNMode_t mode, bool is_bidirec,
                 bool is_test)
      : seq_length_(seq_length),
        batch_size_(batch_size),
        input_size_(input_size),
        hidden_size_(hidden_size),
        num_layers_(num_layers),
        dropout_prob_(dropout_prob),
        seed_(seed),
        weight_numel_(weight_numel),
        mode_(mode),
        is_bidirec_(is_bidirec),
        is_test_(is_test) {}

  template <typename T>
  void Create(const cudnnHandle_t &handle, const platform::Place &place,
              const std::vector<int> &sequence_length, size_t *workspace_size,
              size_t *reserve_size, framework::Tensor *dropout_state) {
    int numDirections = is_bidirec_ ? 2 : 1;
    cudnnDataType_t cudnn_type = platform::CudnnDataType<T>::type;

    // ------------------- cudnn x, y descriptors ---------------------
    std::vector<int> dims_x = {batch_size_, input_size_, 1};
    std::vector<int> strides_x = {input_size_, 1, 1};
    std::vector<int> dims_y = {batch_size_, hidden_size_ * numDirections, 1};
    std::vector<int> strides_y = {hidden_size_ * numDirections, 1, 1};
    for (int i = 0; i < seq_length_; ++i) {
      x_descs_.emplace_back(x_desc_.descriptor<T>(dims_x, strides_x));
      y_descs_.emplace_back(y_desc_.descriptor<T>(dims_y, strides_y));
    }

#if CUDNN_VERSION >= 7201
    if (!sequence_length.empty()) {
      x_seq_desc_.descriptor<T>(seq_length_, batch_size_, input_size_, true,
                                sequence_length);
      y_seq_desc_.descriptor<T>(seq_length_, batch_size_,
                                hidden_size_ * numDirections, true,
                                sequence_length);
    }
#endif

    // ------------------- cudnn hx, hy, cx, cy descriptors----------
    std::vector<int> dims_hx = {num_layers_ * numDirections, batch_size_,
                                hidden_size_};
    std::vector<int> strides_hx = {hidden_size_ * batch_size_, hidden_size_, 1};
    init_h_desc_.descriptor<T>(dims_hx, strides_hx);
    init_c_desc_.descriptor<T>(dims_hx, strides_hx);
    last_h_desc_.descriptor<T>(dims_hx, strides_hx);
    last_c_desc_.descriptor<T>(dims_hx, strides_hx);

    // ------------------- cudnn dropout descriptors ---------------------
    size_t state_size;
    bool is_initialized = dropout_state->IsInitialized();
    if (!is_test_ && !is_initialized) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDropoutGetStatesSize(handle, &state_size));
      dropout_state->mutable_data<uint8_t>({static_cast<int64_t>(state_size)},
                                           place);
    }
    dropout_desc_.descriptor(handle, place, is_initialized, dropout_prob_,
                             is_test_ ? nullptr : dropout_state, seed_,
                             state_size);

// ------------------- cudnn rnn descriptors ---------------------
#if CUDNN_VERSION >= 6000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor_v6(
        handle, rnn_desc_.desc(), hidden_size_, num_layers_,
        dropout_desc_.desc(), CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, mode_,
        CUDNN_RNN_ALGO_STANDARD, cudnn_type));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNDescriptor(
        rnn_desc_.desc(), hidden_size_, num_layers_, dropout_desc_.desc(),
        CUDNN_LINEAR_INPUT,
        is_bidirec_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, mode_,
        cudnn_type));
#endif

#if CUDNN_VERSION >= 7201
    if (!sequence_length.empty()) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetRNNPaddingMode(
          rnn_desc_.desc(), CUDNN_RNN_PADDED_IO_ENABLED));
    }
#endif

    // ------------------- cudnn weights_size ---------------------
    size_t weights_size_;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNParamsSize(
        handle, rnn_desc_.desc(), x_descs_[0], &weights_size_, cudnn_type));
    PADDLE_ENFORCE_EQ(
        weights_size_, sizeof(T) * weight_numel_,
        platform::errors::InvalidArgument(
            "The cudnn rnn and setting weight size should be same."));
    // ------------------- cudnn weight descriptors ---------------------
    platform::DataLayout layout = platform::DataLayout::kNCHW;
    int dim_tmp = weights_size_ / sizeof(T);
    std::vector<int> dim_w = {dim_tmp, 1, 1};
    weight_desc_.descriptor<T>(layout, dim_w);
    // ------------------- cudnn workspace, reserve size ---------------------
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetRNNWorkspaceSize(
        handle, rnn_desc_.desc(), seq_length_, x_descs_.data(),
        workspace_size));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnGetRNNTrainingReserveSize(
            handle, rnn_desc_.desc(), seq_length_, x_descs_.data(),
            reserve_size));
  }
  cudnnTensorDescriptor_t *x_descs() { return x_descs_.data(); }
  cudnnTensorDescriptor_t *y_descs() { return y_descs_.data(); }
#if CUDNN_VERSION >= 7201
  cudnnRNNDataDescriptor_t x_seq_desc() { return x_seq_desc_.desc(); }
  cudnnRNNDataDescriptor_t y_seq_desc() { return y_seq_desc_.desc(); }
#endif
  cudnnTensorDescriptor_t init_h_desc() { return init_h_desc_.desc(); }
  cudnnTensorDescriptor_t init_c_desc() { return init_c_desc_.desc(); }
  cudnnTensorDescriptor_t last_h_desc() { return last_h_desc_.desc(); }
  cudnnTensorDescriptor_t last_c_desc() { return last_c_desc_.desc(); }
  cudnnRNNDescriptor_t rnn_desc() { return rnn_desc_.desc(); }
  cudnnDropoutDescriptor_t dropout_desc() { return dropout_desc_.desc(); }
  cudnnFilterDescriptor_t weight_desc() { return weight_desc_.desc(); }

 private:
  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;
  int num_layers_;
  float dropout_prob_;
  int seed_;
  int weight_numel_;
  cudnnRNNMode_t mode_;
  bool is_bidirec_;
  bool is_test_;
  std::vector<cudnnTensorDescriptor_t> x_descs_;
  std::vector<cudnnTensorDescriptor_t> y_descs_;

  platform::ScopedTensorDescriptor x_desc_;
  platform::ScopedTensorDescriptor y_desc_;
#if CUDNN_VERSION >= 7201
  platform::ScopedRNNTensorDescriptor x_seq_desc_;
  platform::ScopedRNNTensorDescriptor y_seq_desc_;
#endif
  platform::ScopedTensorDescriptor init_h_desc_;
  platform::ScopedTensorDescriptor init_c_desc_;
  platform::ScopedTensorDescriptor last_h_desc_;
  platform::ScopedTensorDescriptor last_c_desc_;
  platform::ScopedDropoutDescriptor dropout_desc_;
  platform::ScopedFilterDescriptor weight_desc_;
  platform::ScopedRNNDescriptor rnn_desc_;
};

template <typename T, typename Type>
bool is_continuous(const Type &weight_list) {
  bool continuous = true;
  for (size_t i = 0; i < weight_list.size() - 1; ++i) {
    auto *in_data = weight_list[i]->template data<T>();
    auto *in_after_data = weight_list[i + 1]->template data<T>();
    auto in_size = weight_list[i]->numel();
    bool temp = in_data + in_size == in_after_data;
    continuous = continuous && temp;
  }
  return continuous;
}

template <typename T>
void weight_to_tensor(const platform::Place &place, cudaStream_t stream,
                      const std::vector<const Tensor *> &weight_list,
                      Tensor *weight) {
  auto weight_data = weight->data<T>();
  int weight_offset = 0;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    const T *in_data = weight_list[i]->data<T>();
    auto in_size = weight_list[i]->numel();

    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, weight->place()),
                 weight_data + weight_offset,
                 BOOST_GET_CONST(platform::CUDAPlace, weight_list[i]->place()),
                 in_data, in_size * sizeof(T), stream);
    weight_offset += in_size;
  }
}

template <typename T>
void weight_to_tensor_list(const platform::Place &place, cudaStream_t stream,
                           std::vector<Tensor *> *weight_grad,
                           const std::vector<const Tensor *> &weight_input,
                           const Tensor *weight) {
  int weight_offset = 0;
  auto *weight_data = weight->data<T>();
  for (size_t i = 0; i < weight_input.size(); ++i) {
    auto in_size = weight_input[i]->numel();
    T *weight_grad_data = (*weight_grad)[i]->mutable_data<T>(place);
    const T *src = weight_data + weight_offset;

    memory::Copy(
        BOOST_GET_CONST(platform::CUDAPlace, (*weight_grad)[i]->place()),
        weight_grad_data, BOOST_GET_CONST(platform::CUDAPlace, weight->place()),
        src, in_size * sizeof(T), stream);
    weight_offset += in_size;
  }
}

template <typename T>
class RNNCudnnKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const Tensor *x = ctx.Input<Tensor>("Input");
    auto pre_state = ctx.MultiInput<Tensor>("PreState");

    Tensor *out = ctx.Output<Tensor>("Out");
    auto state = ctx.MultiOutput<Tensor>("State");
    Tensor *reserve = ctx.Output<Tensor>("Reserve");
    Tensor *state_out = ctx.Output<Tensor>("DropoutState");

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    auto mode = ctx.Attr<std::string>("mode");
    cudnnRNNMode_t rnn_mode = CUDNN_LSTM;
    if (mode == "LSTM")
      rnn_mode = CUDNN_LSTM;
    else if (mode == "GRU")
      rnn_mode = CUDNN_GRU;
    else if (mode == "RNN_RELU")
      rnn_mode = CUDNN_RNN_RELU;
    else if (mode == "RNN_TANH")
      rnn_mode = CUDNN_RNN_TANH;
    else
      PADDLE_THROW(platform::errors::InvalidArgument(
          "rnn_mode should be LSTM, GRU, RNN_RELU or RNN_TANH, but received: "
          "%s.",
          mode));

    bool is_test = ctx.Attr<bool>("is_test");
    int seed = ctx.Attr<int>("seed");
    if (!is_test) {
      int device_id =
          BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace()).GetDeviceId();
      auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);
      if (gen_cuda->GetIsInitPy() && seed == 0) {
        // If perform `manual_seed` in python and inner seed is not specified
        // (equals 0), use global generator generated seed.
        seed = static_cast<int>(gen_cuda->Random64());
      } else if (seed == 0) {
        // use random generated seed
        std::random_device rd;
        seed = rd();
      }  // else use `ctx.Attr<int>("seed")` specified seed
    }

    const T *x_data = x->data<T>();
    const T *init_h_data = pre_state[0]->data<T>();
    const T *init_c_data = nullptr;
    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    T *last_h_data = state[0]->mutable_data<T>(ctx.GetPlace());
    T *last_c_data = nullptr;
    if (rnn_mode == CUDNN_LSTM) {
      init_c_data = pre_state[1]->data<T>();
      last_c_data = state[1]->mutable_data<T>(ctx.GetPlace());
    }

    bool has_seq_length = ctx.HasInput("SequenceLength");
    std::vector<int> SequenceLength;
    if (has_seq_length) {
      auto *sequence_length = ctx.Input<Tensor>("SequenceLength");
      SequenceLength = operators::GetDataFromTensor<int>(sequence_length);
    }

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    int seq_length = x->dims()[0];
    int batch_size = x->dims()[1];
    int input_size = x->dims()[2];

    size_t workspace_size;
    size_t reserve_size;
    Tensor weight_whole;
    T *w_data = nullptr;
    auto place = ctx.GetPlace();
    auto stream = reinterpret_cast<const platform::CUDADeviceContext &>(
                      ctx.device_context())
                      .stream();
    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    auto weight_numel = std::accumulate(
        weight_list.begin(), weight_list.end(), 0,
        [](int64_t num, const Tensor *t) { return num + t->numel(); });
    bool continuous =
        is_continuous<T, std::vector<const Tensor *>>(weight_list);
    if (!continuous) {
      LOG_FIRST_N(WARNING, 2)
          << "If the memory space of the Input WeightList is not continuous, "
             "less efficient calculation will be called. Please call "
             "flatten_parameters() to make the input memory continuous.";
      weight_whole.mutable_data<T>({weight_numel}, place);
      weight_to_tensor<T>(place, stream, weight_list, &weight_whole);
      w_data = weight_whole.data<T>();
      if (is_test) {  // maybe also reset small weights' ptr for training
        int offset = 0;
        for (size_t i = 0; i < weight_list.size(); ++i) {
          size_t len = weight_list[i]->numel();
          auto dim = weight_list[i]->dims();
          const_cast<Tensor *>(weight_list[i])
              ->ShareDataWith(
                  weight_whole.Slice(static_cast<int64_t>(offset),
                                     static_cast<int64_t>(offset + len)))
              .Resize(dim);
          offset += len;
        }
      }
    } else {
      w_data = const_cast<T *>(weight_list[0]->data<T>());
    }

    RNNDescriptors rnn(seq_length, batch_size, input_size, hidden_size,
                       num_layers, dropout_prob, seed, weight_numel, rnn_mode,
                       is_bidirec, is_test);
    rnn.Create<T>(handle, ctx.GetPlace(), SequenceLength, &workspace_size,
                  &reserve_size, state_out);

    framework::Tensor workspace_data_;
    workspace_data_.mutable_data<uint8_t>(
        {static_cast<int64_t>(workspace_size)}, ctx.GetPlace());

    auto *reserve_data = reserve->mutable_data<uint8_t>(
        {static_cast<int64_t>(reserve_size)}, ctx.GetPlace());

    if (is_test) {
      RNNInferece(has_seq_length, handle, seq_length, &rnn, x_data, init_h_data,
                  init_c_data, w_data, out_data, last_h_data, last_c_data,
                  &workspace_data_, workspace_size);
    } else {
      if (!has_seq_length) {
        // for train
        // This interface is used when the input/output is unpadded.
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardTraining(
            handle, rnn.rnn_desc(), seq_length, rnn.x_descs(), x_data,
            rnn.init_h_desc(), init_h_data, rnn.init_c_desc(), init_c_data,
            rnn.weight_desc(), w_data, rnn.y_descs(), out_data,
            rnn.last_h_desc(), last_h_data, rnn.last_c_desc(), last_c_data,
            workspace_data_.data<uint8_t>(), workspace_size, reserve_data,
            reserve_size));
      } else {
#if CUDNN_VERSION >= 7201
        // for train
        // This interface is used when the input/output is padded.
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnRNNForwardTrainingEx(
                handle, rnn.rnn_desc(), rnn.x_seq_desc(), x_data,
                rnn.init_h_desc(), init_h_data, rnn.init_c_desc(), init_c_data,
                rnn.weight_desc(), w_data, rnn.y_seq_desc(), out_data,
                rnn.last_h_desc(), last_h_data, rnn.last_c_desc(), last_c_data,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, workspace_data_.data<uint8_t>(), workspace_size,
                reserve_data, reserve_size));
#else
        PADDLE_THROW(platform::errors::Unavailable(
            "The padded input is supported by "
            "cudnnRNNForwardTrainingEx, but it only works when "
            "the version of cudnn is larger than 7.2.1"));
#endif
      }
    }
  }

  void RNNInferece(const bool &has_seq_length, const cudnnHandle_t &handle,
                   const int &seq_length, RNNDescriptors *rnn, const T *x_data,
                   const T *init_h_data, const T *init_c_data, const T *w_data,
                   T *out_data, T *last_h_data, T *last_c_data,
                   framework::Tensor *workspace_data,
                   const size_t &workspace_size) const {
    if (!has_seq_length) {
      // for inference
      // This interface is used when the input/output is unpadded.
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardInference(
          handle, rnn->rnn_desc(), seq_length, rnn->x_descs(), x_data,
          rnn->init_h_desc(), init_h_data, rnn->init_c_desc(), init_c_data,
          rnn->weight_desc(), w_data, rnn->y_descs(), out_data,
          rnn->last_h_desc(), last_h_data, rnn->last_c_desc(), last_c_data,
          workspace_data->data<uint8_t>(), workspace_size));
    } else {
#if CUDNN_VERSION >= 7201
      // for inference
      // This interface is used when the input/output is padded.
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNForwardInferenceEx(
          handle, rnn->rnn_desc(), rnn->x_seq_desc(), x_data,
          rnn->init_h_desc(), init_h_data, rnn->init_c_desc(), init_c_data,
          rnn->weight_desc(), w_data, rnn->y_seq_desc(), out_data,
          rnn->last_h_desc(), last_h_data, rnn->last_c_desc(), last_c_data,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, workspace_data->data<uint8_t>(), workspace_size));
#else
      // CUDNN VERSION has to >=7.2.1
      PADDLE_THROW(platform::errors::Unavailable(
          "The padded input is supported by "
          "cudnnRNNForwardInferenceEx, but it only works when "
          "the version of cudnn is larger than 7.2.1"));
#endif
    }
  }
};

template <typename T>
class RNNGradCudnnKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input = ctx.Input<Tensor>("Input");
    auto pre_state = ctx.MultiInput<Tensor>("PreState");
    auto weight_list = ctx.MultiInput<Tensor>("WeightList");
    auto *state_out = ctx.Input<Tensor>("DropoutState");
    auto *reserve = ctx.Input<Tensor>("Reserve");
    auto *out = ctx.Input<Tensor>("Out");
    // auto state = ctx.MultiInput<Tensor>("State");

    auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto state_grad = ctx.MultiInput<Tensor>(framework::GradVarName("State"));

    auto *in_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto pre_state_grad =
        ctx.MultiOutput<Tensor>(framework::GradVarName("PreState"));
    auto weight_grad_list =
        ctx.MultiOutput<Tensor>(framework::GradVarName("WeightList"));

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    auto mode = ctx.Attr<std::string>("mode");
    cudnnRNNMode_t rnn_mode = CUDNN_LSTM;
    if (mode == "LSTM")
      rnn_mode = CUDNN_LSTM;
    else if (mode == "GRU")
      rnn_mode = CUDNN_GRU;
    else if (mode == "RNN_RELU")
      rnn_mode = CUDNN_RNN_RELU;
    else if (mode == "RNN_TANH")
      rnn_mode = CUDNN_RNN_TANH;
    else
      PADDLE_THROW(platform::errors::InvalidArgument(
          "rnn_mode should be LSTM, GRU, RNN_RELU or RNN_TANH, but received: "
          "%s.",
          mode));
    bool is_test = ctx.Attr<bool>("is_test");
    int seed = ctx.Attr<int>("seed");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    auto place = ctx.GetPlace();
    auto weight_numel = std::accumulate(
        weight_list.begin(), weight_list.end(), 0,
        [](int64_t num, const Tensor *t) { return num + t->numel(); });
    bool continuous =
        is_continuous<T, std::vector<const Tensor *>>(weight_list);

    auto stream = reinterpret_cast<const platform::CUDADeviceContext &>(
                      ctx.device_context())
                      .stream();
    Tensor weight_whole;
    T *weight_data = nullptr;

    if (!continuous) {
      weight_whole.mutable_data<T>({weight_numel}, place);
      weight_to_tensor<T>(place, stream, weight_list, &weight_whole);
      weight_data = weight_whole.data<T>();
    } else {
      weight_data = const_cast<T *>(weight_list[0]->data<T>());
    }

    Tensor weight_grad;
    math::SetConstant<paddle::platform::CUDADeviceContext, T> zero;
    weight_grad.mutable_data<T>({weight_numel}, ctx.GetPlace());
    zero(dev_ctx, &weight_grad, static_cast<T>(0.0));
    T *weight_grad_data = weight_grad.data<T>();

    int offset = 0;
    for (size_t i = 0; i < weight_grad_list.size(); ++i) {
      size_t len = weight_grad_list[i]->numel();
      auto dim = weight_grad_list[i]->dims();
      weight_grad_list[i]
          ->ShareDataWith(weight_grad.Slice(static_cast<int64_t>(offset),
                                            static_cast<int64_t>(offset + len)))
          .Resize(dim);
      offset += len;
    }

    Tensor input_grad_value;
    if (!in_grad) {
      in_grad = &input_grad_value;
      in_grad->Resize(input->dims());
    }

    auto *init_h_data = pre_state[0]->data<T>();
    // auto *last_h_data = state[0]->data<T>();
    auto *last_h_grad_data = state_grad[0]->data<T>();
    const T *init_c_data = nullptr;
    // const T *last_c_data = nullptr;
    const T *last_c_grad_data = nullptr;
    T *init_h_grad_data =
        pre_state_grad.size() != 0 && pre_state_grad[0]
            ? pre_state_grad[0]->mutable_data<T>(ctx.GetPlace())
            : nullptr;
    T *init_c_grad_data = nullptr;
    if (rnn_mode == CUDNN_LSTM) {
      init_c_data = pre_state[1]->data<T>();
      // last_c_data = state[1]->data<T>();
      last_c_grad_data = state_grad[1]->data<T>();
      init_c_grad_data =
          pre_state_grad.size() != 0 && pre_state_grad[1]
              ? pre_state_grad[1]->mutable_data<T>(ctx.GetPlace())
              : nullptr;
    }
    auto *out_data = out->data<T>();
    auto *out_grad_data = out_grad->data<T>();
    // maybe need check exist
    auto *in_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());

    bool has_seq_length = ctx.HasInput("SequenceLength");
    std::vector<int> SequenceLength;
    if (has_seq_length) {
      auto *sequence_length = ctx.Input<Tensor>("SequenceLength");
      SequenceLength = operators::GetDataFromTensor<int>(sequence_length);
    }

    auto input_dims = input->dims();
    int seq_length = input_dims[0];
    int batch_size = input_dims[1];
    int input_size = input_dims[2];

    size_t workspace_size;
    size_t reserve_size;

    RNNDescriptors rnn(seq_length, batch_size, input_size, hidden_size,
                       num_layers, dropout_prob, seed, weight_numel, rnn_mode,
                       is_bidirec, is_test);

    rnn.Create<T>(handle, ctx.GetPlace(), SequenceLength, &workspace_size,
                  &reserve_size, const_cast<Tensor *>(state_out));

    framework::Tensor workspace_data_;
    workspace_data_.mutable_data<uint8_t>(
        {static_cast<int64_t>(workspace_size)}, ctx.GetPlace());
    const uint8_t *reserve_data = reserve->data<uint8_t>();

    if (!has_seq_length) {
      // This interface is used when the input/output is unpadded.
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardData(
          handle, rnn.rnn_desc(), seq_length, rnn.y_descs(), out_data,
          rnn.y_descs(), out_grad_data, rnn.last_h_desc(), last_h_grad_data,
          rnn.last_c_desc(), last_c_grad_data, rnn.weight_desc(), weight_data,
          rnn.init_h_desc(), init_h_data, rnn.init_c_desc(), init_c_data,
          rnn.x_descs(), in_grad_data, rnn.init_h_desc(), init_h_grad_data,
          rnn.init_c_desc(), init_c_grad_data, workspace_data_.data<uint8_t>(),
          workspace_size, const_cast<uint8_t *>(reserve_data), reserve_size));

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardWeights(
          handle, rnn.rnn_desc(), seq_length, rnn.x_descs(), input->data<T>(),
          rnn.init_h_desc(), init_h_data, rnn.y_descs(), out->data<T>(),
          workspace_data_.data<uint8_t>(), workspace_size, rnn.weight_desc(),
          weight_grad_data, const_cast<uint8_t *>(reserve_data), reserve_size));
    } else {
#if CUDNN_VERSION >= 7201
      // for train
      // This interface is used when the input/output is padded.
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardDataEx(
          handle, rnn.rnn_desc(), rnn.y_seq_desc(), out_data, rnn.y_seq_desc(),
          out_grad_data, nullptr, nullptr, rnn.last_h_desc(), last_h_grad_data,
          rnn.last_c_desc(), last_c_grad_data, rnn.weight_desc(), weight_data,
          rnn.init_h_desc(), init_h_data, rnn.init_c_desc(), init_c_data,
          rnn.x_seq_desc(), in_grad_data, rnn.init_h_desc(), init_h_grad_data,
          rnn.init_c_desc(), init_c_grad_data, nullptr, nullptr,
          workspace_data_.data<uint8_t>(), workspace_size,
          const_cast<uint8_t *>(reserve_data), reserve_size));

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnRNNBackwardWeightsEx(
          handle, rnn.rnn_desc(), rnn.x_seq_desc(), input->data<T>(),
          rnn.init_h_desc(), init_h_data, rnn.y_seq_desc(), out->data<T>(),
          workspace_data_.data<uint8_t>(), workspace_size, rnn.weight_desc(),
          weight_grad_data, const_cast<uint8_t *>(reserve_data), reserve_size));
#else
      PADDLE_THROW(platform::errors::Unavailable(
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
REGISTER_OP_CUDA_KERNEL(rnn, ops::RNNCudnnKernel<float>,
                        ops::RNNCudnnKernel<double>);
REGISTER_OP_CUDA_KERNEL(rnn_grad, ops::RNNGradCudnnKernel<float>,
                        ops::RNNGradCudnnKernel<double>);
