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
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/operators/cudnn_lstm_cache.h"
#endif
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/miopen_lstm_cache.h"
#endif

namespace paddle {
namespace operators {

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

int size_sum(const std::vector<const phi::DenseTensor *> &weight_list) {
  int size = 0;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    auto in_size = weight_list[i]->numel();
    size += in_size;
  }
  return size;
}

template <typename T>
void weight_to_tensor(const platform::Place &place,
                      gpuStream_t stream,
                      const std::vector<const phi::DenseTensor *> &weight_list,
                      phi::DenseTensor *weight) {
  auto weight_data = weight->data<T>();
  int weight_offset = 0;
  for (size_t i = 0; i < weight_list.size(); ++i) {
    const T *in_data = weight_list[i]->data<T>();
    auto in_size = weight_list[i]->numel();

    memory::Copy(weight->place(),
                 weight_data + weight_offset,
                 weight_list[i]->place(),
                 in_data,
                 in_size * sizeof(T),
                 stream);
    weight_offset += in_size;
  }
}

template <typename T>
void weight_to_tensor_list(
    const platform::Place &place,
    gpuStream_t stream,
    std::vector<phi::DenseTensor *> *weight_grad,
    const std::vector<const phi::DenseTensor *> &weight_input,
    const phi::DenseTensor *weight) {
  int weight_offset = 0;
  auto *weight_data = weight->data<T>();
  for (size_t i = 0; i < weight_input.size(); ++i) {
    auto in_size = weight_input[i]->numel();
    T *weight_grad_data = (*weight_grad)[i]->mutable_data<T>(place);
    const T *src = weight_data + weight_offset;

    memory::Copy((*weight_grad)[i]->place(),
                 weight_grad_data,
                 weight->place(),
                 src,
                 in_size * sizeof(T),
                 stream);
    weight_offset += in_size;
  }
}

template <typename T>
#ifdef PADDLE_WITH_HIP
void LSTMInferece(const bool &has_seq_length,
                  const miopenHandle_t &handle,
#else
void LSTMInferece(const bool &has_seq_length,
                  const cudnnHandle_t &handle,
#endif
                  const int &seq_length,
                  ScopedRNNBase *rnn,
                  const T *x_data,
                  const T *init_h_data,
                  const T *init_c_data,
                  const T *w_data,
                  T *out_data,
                  T *last_h_data,
                  T *last_c_data,
                  phi::DenseTensor *workspace_data,
                  const size_t &workspace_size) {
  if (!has_seq_length) {
// for inference
// This interface is used when the input/output is unpadded.
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenRNNForwardInference(
        handle,
        rnn->rnn_desc(),
        seq_length,
        rnn->x_descs(),
        x_data,
        rnn->init_h_desc(),
        init_h_data,
        rnn->init_c_desc(),
        init_c_data,
        rnn->weight_desc(),
        w_data,
        rnn->y_descs(),
        out_data,
        rnn->last_h_desc(),
        last_h_data,
        rnn->last_c_desc(),
        last_c_data,
        workspace_data->data<uint8_t>(),
        workspace_size));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnRNNForwardInference(
        handle,
        rnn->rnn_desc(),
        seq_length,
        rnn->x_descs(),
        x_data,
        rnn->init_h_desc(),
        init_h_data,
        rnn->init_c_desc(),
        init_c_data,
        rnn->weight_desc(),
        w_data,
        rnn->y_descs(),
        out_data,
        rnn->last_h_desc(),
        last_h_data,
        rnn->last_c_desc(),
        last_c_data,
        workspace_data->data<uint8_t>(),
        workspace_size));
#endif
  } else {
#if !defined(PADDLE_WITH_HIP) && CUDNN_VERSION >= 7201
    // for inference
    // This interface is used when the input/output is padded.
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnRNNForwardInferenceEx(
        handle,
        rnn->rnn_desc(),
        rnn->x_seq_desc(),
        x_data,
        rnn->init_h_desc(),
        init_h_data,
        rnn->init_c_desc(),
        init_c_data,
        rnn->weight_desc(),
        w_data,
        rnn->y_seq_desc(),
        out_data,
        rnn->last_h_desc(),
        last_h_data,
        rnn->last_c_desc(),
        last_c_data,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        workspace_data->data<uint8_t>(),
        workspace_size));
#else
    // CUDNN VERSION has to >=7.2.1
    PADDLE_THROW(platform::errors::Unavailable(
        "The padded input is supported by "
        "cudnnRNNForwardInferenceEx, but it only works when "
        "the version of cudnn is larger than 7.2.1"));
#endif
  }
}

template <typename T>
class CudnnLSTMGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const phi::DenseTensor *x = ctx.Input<phi::DenseTensor>("Input");
    const phi::DenseTensor *init_h = ctx.Input<phi::DenseTensor>("InitH");
    const phi::DenseTensor *init_c = ctx.Input<phi::DenseTensor>("InitC");

    phi::DenseTensor *out = ctx.Output<phi::DenseTensor>("Out");
    phi::DenseTensor *last_h = ctx.Output<phi::DenseTensor>("LastH");
    phi::DenseTensor *last_c = ctx.Output<phi::DenseTensor>("LastC");
    phi::DenseTensor *reserve = ctx.Output<phi::DenseTensor>("Reserve");
    phi::DenseTensor *state_out = ctx.Output<phi::DenseTensor>("StateOut");

    const T *x_data = x->data<T>();
    const T *init_h_data = init_h->data<T>();
    const T *init_c_data = init_c->data<T>();

    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    T *last_h_data = last_h->mutable_data<T>(ctx.GetPlace());
    T *last_c_data = last_c->mutable_data<T>(ctx.GetPlace());

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    bool is_test = ctx.Attr<bool>("is_test");
    int seed = ctx.Attr<int>("seed");

    if (!is_test) {
      if (seed == 0) {
        // If not specify seed, use global Generator to generate seed.
        int device_id = ctx.GetPlace().GetDeviceId();
        auto gen_cuda = phi::DefaultCUDAGenerator(device_id);
        seed = static_cast<int>(gen_cuda->Random64());
      }
      // else use `ctx.Attr<int>("seed")` specified seed
    }

    bool has_seq_length = ctx.HasInput("SequenceLength");
    std::vector<int> SequenceLength;
    if (has_seq_length) {
      auto *sequence_length = ctx.Input<phi::DenseTensor>("SequenceLength");
      SequenceLength = operators::GetDataFromTensor<int>(sequence_length);
    }

    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto handle = dev_ctx.cudnn_handle();

    int seq_length = x->dims()[0];
    int batch_size = x->dims()[1];
    int input_size = x->dims()[2];
    bool state_initialized = state_out->IsInitialized() ? true : false;

    size_t workspace_size;
    size_t reserve_size;
    phi::DenseTensor weight_whole;
    T *w_data = nullptr;
    int weight_numel;
    bool w_initialized = false;
    auto place = ctx.GetPlace();
    auto stream =
        reinterpret_cast<const phi::GPUContext &>(ctx.device_context())
            .stream();
    if (is_test && ctx.HasInput("W")) {
      auto *W = ctx.Input<phi::DenseTensor>("W");
      w_initialized = W->IsInitialized() ? true : false;
      weight_numel = W->numel();
    }
    if (!w_initialized) {
      auto weight_list = ctx.MultiInput<phi::DenseTensor>("WeightList");
      bool continuous =
          is_continuous<T, std::vector<const phi::DenseTensor *>>(weight_list);
      weight_numel = size_sum(weight_list);

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
            const_cast<phi::DenseTensor *>(weight_list[i])
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
    } else {
      auto *W = ctx.Input<phi::DenseTensor>("W");
      w_data = const_cast<T *>(W->data<T>());
    }

    ScopedRNNBase rnn(seq_length,
                      batch_size,
                      input_size,
                      hidden_size,
                      num_layers,
                      dropout_prob,
                      seed,
                      weight_numel,
                      state_initialized,
                      is_bidirec);
    rnn.Create<T>(handle,
                  ctx.GetPlace(),
                  SequenceLength,
                  &workspace_size,
                  &reserve_size,
                  state_out);

    phi::DenseTensor workspace_data_;
    workspace_data_.mutable_data<uint8_t>(
        {static_cast<int64_t>(workspace_size)}, ctx.GetPlace());

    auto *reserve_data = reserve->mutable_data<uint8_t>(
        {static_cast<int64_t>(reserve_size)}, ctx.GetPlace());

    if (is_test) {
      LSTMInferece<T>(has_seq_length,
                      handle,
                      seq_length,
                      &rnn,
                      x_data,
                      init_h_data,
                      init_c_data,
                      w_data,
                      out_data,
                      last_h_data,
                      last_c_data,
                      &workspace_data_,
                      workspace_size);
    } else {
      if (!has_seq_length) {
// for train
// This interface is used when the input/output is unpadded.
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenRNNForwardTraining(
            handle,
            rnn.rnn_desc(),
            seq_length,
            rnn.x_descs(),
            x_data,
            rnn.init_h_desc(),
            init_h_data,
            rnn.init_c_desc(),
            init_c_data,
            rnn.weight_desc(),
            w_data,
            rnn.y_descs(),
            out_data,
            rnn.last_h_desc(),
            last_h_data,
            rnn.last_c_desc(),
            last_c_data,
            workspace_data_.data<uint8_t>(),
            workspace_size,
            reserve_data,
            reserve_size));
#else
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnRNNForwardTraining(
            handle,
            rnn.rnn_desc(),
            seq_length,
            rnn.x_descs(),
            x_data,
            rnn.init_h_desc(),
            init_h_data,
            rnn.init_c_desc(),
            init_c_data,
            rnn.weight_desc(),
            w_data,
            rnn.y_descs(),
            out_data,
            rnn.last_h_desc(),
            last_h_data,
            rnn.last_c_desc(),
            last_c_data,
            workspace_data_.data<uint8_t>(),
            workspace_size,
            reserve_data,
            reserve_size));
#endif
      } else {
#if !defined(PADDLE_WITH_HIP) && CUDNN_VERSION >= 7201
        // for train
        // This interface is used when the input/output is padded.
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnRNNForwardTrainingEx(
            handle,
            rnn.rnn_desc(),
            rnn.x_seq_desc(),
            x_data,
            rnn.init_h_desc(),
            init_h_data,
            rnn.init_c_desc(),
            init_c_data,
            rnn.weight_desc(),
            w_data,
            rnn.y_seq_desc(),
            out_data,
            rnn.last_h_desc(),
            last_h_data,
            rnn.last_c_desc(),
            last_c_data,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            workspace_data_.data<uint8_t>(),
            workspace_size,
            reserve_data,
            reserve_size));
#else
        PADDLE_THROW(platform::errors::Unavailable(
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
    auto *input = ctx.Input<phi::DenseTensor>("Input");
    auto *init_h = ctx.Input<phi::DenseTensor>("InitH");
    auto *init_c = ctx.Input<phi::DenseTensor>("InitC");
    auto *reserve = ctx.Input<phi::DenseTensor>("Reserve");
    auto *state_out = ctx.Input<phi::DenseTensor>("StateOut");
    auto weight_list = ctx.MultiInput<phi::DenseTensor>("WeightList");

    auto *out = ctx.Input<phi::DenseTensor>("Out");
    auto *out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *last_h_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("LastH"));
    auto *last_c_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("LastC"));

    auto *in_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));
    auto *init_h_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("InitH"));
    auto *init_c_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("InitC"));
    auto weight_grad_list =
        ctx.MultiOutput<phi::DenseTensor>(framework::GradVarName("WeightList"));

    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto handle = dev_ctx.cudnn_handle();

    auto input_dims = input->dims();
    auto init_h_dims = init_h->dims();
    auto init_c_dims = init_c->dims();

    auto *init_h_data = init_h->data<T>();
    auto *init_c_data = init_c->data<T>();
    auto *out_data = out->data<T>();
    auto *out_grad_data = out_grad->data<T>();
    auto *last_h_grad_data = last_h_grad->data<T>();
    auto *last_c_grad_data = last_c_grad->data<T>();

    auto place = ctx.GetPlace();
    int weight_numel = size_sum(weight_list);
    bool continuous =
        is_continuous<T, std::vector<const phi::DenseTensor *>>(weight_list);

    auto stream =
        reinterpret_cast<const phi::GPUContext &>(ctx.device_context())
            .stream();
    phi::DenseTensor weight_whole;
    T *weight_data = nullptr;

    if (!continuous) {
      weight_whole.mutable_data<T>({weight_numel}, place);
      weight_to_tensor<T>(place, stream, weight_list, &weight_whole);
      weight_data = weight_whole.data<T>();
    } else {
      weight_data = const_cast<T *>(weight_list[0]->data<T>());
    }

    phi::DenseTensor weight_grad;
    phi::funcs::SetConstant<phi::GPUContext, T> zero;
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

    in_grad->mutable_data<T>(input_dims, ctx.GetPlace());
    auto *in_grad_data = in_grad->data<T>();

    if (init_h_grad) init_h_grad->mutable_data<T>(init_h_dims, ctx.GetPlace());
    auto *init_h_grad_data = init_h_grad ? init_h_grad->data<T>() : nullptr;

    if (init_c_grad) init_c_grad->mutable_data<T>(init_c_dims, ctx.GetPlace());
    auto *init_c_grad_data = init_c_grad ? init_c_grad->data<T>() : nullptr;

    float dropout_prob = ctx.Attr<float>("dropout_prob");
    bool is_bidirec = ctx.Attr<bool>("is_bidirec");
    int hidden_size = ctx.Attr<int>("hidden_size");
    int num_layers = ctx.Attr<int>("num_layers");
    int seed = ctx.Attr<int>("seed");

    bool has_seq_length = ctx.HasInput("SequenceLength");
    std::vector<int> SequenceLength;
    if (has_seq_length) {
      auto *sequence_length = ctx.Input<phi::DenseTensor>("SequenceLength");
      SequenceLength = operators::GetDataFromTensor<int>(sequence_length);
    }

    int seq_length = input_dims[0];
    int batch_size = input->dims()[1];
    int input_size = input->dims()[2];

    size_t workspace_size;
    size_t reserve_size;

    ScopedRNNBase rnn(seq_length,
                      batch_size,
                      input_size,
                      hidden_size,
                      num_layers,
                      dropout_prob,
                      seed,
                      weight_numel,
                      true,
                      is_bidirec);

    rnn.Create<T>(handle,
                  ctx.GetPlace(),
                  SequenceLength,
                  &workspace_size,
                  &reserve_size,
                  const_cast<phi::DenseTensor *>(state_out));

    phi::DenseTensor workspace_data_;
    workspace_data_.mutable_data<uint8_t>(
        {static_cast<int64_t>(workspace_size)}, ctx.GetPlace());
    const uint8_t *reserve_data = reserve->data<uint8_t>();

    if (!has_seq_length) {
// This interface is used when the input/output is unpadded.
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenRNNBackwardData(
          handle,
          rnn.rnn_desc(),
          seq_length,
          rnn.y_descs(),
          out_data,
          rnn.y_descs(),
          out_grad_data,
          rnn.last_h_desc(),
          last_h_grad_data,
          rnn.last_c_desc(),
          last_c_grad_data,
          rnn.weight_desc(),
          weight_data,
          rnn.init_h_desc(),
          init_h_data,
          rnn.init_c_desc(),
          init_c_data,
          rnn.x_descs(),
          in_grad_data,
          rnn.init_h_desc(),
          init_h_grad_data,
          rnn.init_c_desc(),
          init_c_grad_data,
          workspace_data_.data<uint8_t>(),
          workspace_size,
          const_cast<uint8_t *>(reserve_data),
          reserve_size));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::miopenRNNBackwardWeights(
          handle,
          rnn.rnn_desc(),
          seq_length,
          rnn.x_descs(),
          input->data<T>(),
          rnn.init_h_desc(),
          init_h->data<T>(),
          rnn.y_descs(),
          out->data<T>(),
          rnn.weight_desc(),
          weight_grad_data,
          workspace_data_.data<uint8_t>(),
          workspace_size,
          const_cast<uint8_t *>(reserve_data),
          reserve_size));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnRNNBackwardData(
          handle,
          rnn.rnn_desc(),
          seq_length,
          rnn.y_descs(),
          out_data,
          rnn.y_descs(),
          out_grad_data,
          rnn.last_h_desc(),
          last_h_grad_data,
          rnn.last_c_desc(),
          last_c_grad_data,
          rnn.weight_desc(),
          weight_data,
          rnn.init_h_desc(),
          init_h_data,
          rnn.init_c_desc(),
          init_c_data,
          rnn.x_descs(),
          in_grad_data,
          rnn.init_h_desc(),
          init_h_grad_data,
          rnn.init_c_desc(),
          init_c_grad_data,
          workspace_data_.data<uint8_t>(),
          workspace_size,
          const_cast<uint8_t *>(reserve_data),
          reserve_size));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnRNNBackwardWeights(
          handle,
          rnn.rnn_desc(),
          seq_length,
          rnn.x_descs(),
          input->data<T>(),
          rnn.init_h_desc(),
          init_h->data<T>(),
          rnn.y_descs(),
          out->data<T>(),
          workspace_data_.data<uint8_t>(),
          workspace_size,
          rnn.weight_desc(),
          weight_grad_data,
          const_cast<uint8_t *>(reserve_data),
          reserve_size));
#endif
    } else {
#if !defined(PADDLE_WITH_HIP) && CUDNN_VERSION >= 7201
      // for train
      // This interface is used when the input/output is padded.
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnRNNBackwardDataEx(
          handle,
          rnn.rnn_desc(),
          rnn.y_seq_desc(),
          out_data,
          rnn.y_seq_desc(),
          out_grad_data,
          nullptr,
          nullptr,
          rnn.last_h_desc(),
          last_h_grad_data,
          rnn.last_c_desc(),
          last_c_grad_data,
          rnn.weight_desc(),
          weight_data,
          rnn.init_h_desc(),
          init_h_data,
          rnn.init_c_desc(),
          init_c_data,
          rnn.x_seq_desc(),
          in_grad_data,
          rnn.init_h_desc(),
          init_h_grad_data,
          rnn.init_c_desc(),
          init_c_grad_data,
          nullptr,
          nullptr,
          workspace_data_.data<uint8_t>(),
          workspace_size,
          const_cast<uint8_t *>(reserve_data),
          reserve_size));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnRNNBackwardWeightsEx(
          handle,
          rnn.rnn_desc(),
          rnn.x_seq_desc(),
          input->data<T>(),
          rnn.init_h_desc(),
          init_h->data<T>(),
          rnn.y_seq_desc(),
          out->data<T>(),
          workspace_data_.data<uint8_t>(),
          workspace_size,
          rnn.weight_desc(),
          weight_grad_data,
          const_cast<uint8_t *>(reserve_data),
          reserve_size));
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
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(cudnn_lstm, ops::CudnnLSTMGPUKernel<float>);
REGISTER_OP_CUDA_KERNEL(cudnn_lstm_grad, ops::CudnnLSTMGPUGradKernel<float>);
#else
REGISTER_OP_CUDA_KERNEL(cudnn_lstm,
                        ops::CudnnLSTMGPUKernel<float>,
                        ops::CudnnLSTMGPUKernel<double>);
REGISTER_OP_CUDA_KERNEL(cudnn_lstm_grad,
                        ops::CudnnLSTMGPUGradKernel<float>,
                        ops::CudnnLSTMGPUGradKernel<double>);
#endif
