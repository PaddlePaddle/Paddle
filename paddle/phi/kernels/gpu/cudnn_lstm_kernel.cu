// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/cudnn_lstm_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/cudnn_lstm_utils.h"

namespace phi {

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
#if CUDNN_VERSION >= 90000
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnRNNForward(handle,
                                    rnn->rnn_desc(),
                                    CUDNN_FWD_MODE_INFERENCE,
                                    nullptr,
                                    rnn->x_seq_desc(),
                                    x_data,
                                    rnn->y_seq_desc(),
                                    out_data,
                                    rnn->init_h_desc(),
                                    init_h_data,
                                    last_h_data,
                                    rnn->init_c_desc(),
                                    init_c_data,
                                    last_c_data,
                                    rnn->weights_size(),
                                    w_data,
                                    workspace_size,
                                    workspace_data->data<uint8_t>(),
                                    0,
                                    nullptr));

#else

  if (!has_seq_length) {
// for inference
// This interface is used when the input/output is unpadded.
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::miopenRNNForwardInference(handle,
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
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnRNNForwardInference(handle,
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnRNNForwardInferenceEx(
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
    PADDLE_THROW(common::errors::Unavailable(
        "The padded input is supported by "
        "cudnnRNNForwardInferenceEx, but it only works when "
        "the version of cudnn is larger than 7.2.1"));
#endif
  }

#endif  // end CUDNN_VERSION >= 90000
}

template <typename T, typename Context>
void CudnnLSTMKernel(
    const Context &ctx,
    const DenseTensor &x,
    const DenseTensor &init_h,
    const DenseTensor &init_c,
    const paddle::optional<DenseTensor> &w,
    const paddle::optional<std::vector<const DenseTensor *>> &weight_list,
    const paddle::optional<DenseTensor> &sequence_length,
    float dropout_prob,
    bool is_bidirec,
    int hidden_size,
    int num_layers,
    bool is_test,
    int seed,
    DenseTensor *out,
    DenseTensor *last_h,
    DenseTensor *last_c,
    DenseTensor *reserve,
    DenseTensor *state_out) {
  const T *x_data = x.data<T>();
  const T *init_h_data = init_h.data<T>();
  const T *init_c_data = init_c.data<T>();

  T *out_data = ctx.template Alloc<T>(out);
  T *last_h_data = ctx.template Alloc<T>(last_h);
  T *last_c_data = ctx.template Alloc<T>(last_c);

  if (!is_test) {
    if (seed == 0) {
      // If not specify seed, use global Generator to generate seed.
      int device_id = ctx.GetPlace().GetDeviceId();
      auto gen_cuda = phi::DefaultCUDAGenerator(device_id);
      seed = static_cast<int>(gen_cuda->Random64());
    }
  }

  auto *running_sequence_length = sequence_length.get_ptr();
  bool has_seq_length = running_sequence_length != nullptr;
  std::vector<int> SequenceLength;
  if (has_seq_length) {
    SequenceLength = phi::GetVectorFromTensor<int>(running_sequence_length);
  }

  auto handle = ctx.cudnn_handle();

  int seq_length = x.dims()[0];
  int batch_size = x.dims()[1];
  int input_size = x.dims()[2];
  bool state_initialized = state_out->initialized() ? true : false;

  size_t workspace_size;
  size_t reserve_size;
  phi::DenseTensor weight_whole;
  T *w_data = nullptr;
  int weight_numel;
  bool w_initialized = false;
  auto place = ctx.GetPlace();
  auto stream = ctx.stream();
  auto *running_w = w.get_ptr();
  if (is_test && running_w != nullptr) {
    w_initialized = running_w->initialized() ? true : false;
    weight_numel = running_w->numel();
  }
  if (!w_initialized) {
    auto running_weight_list = *weight_list.get_ptr();
    bool continuous = is_continuous<T, std::vector<const phi::DenseTensor *>>(
        running_weight_list);
    weight_numel = size_sum(running_weight_list);

    if (!continuous) {
      LOG_FIRST_N(WARNING, 2)
          << "If the memory space of the Input WeightList is not continuous, "
             "less efficient calculation will be called. Please call "
             "flatten_parameters() to make the input memory continuous.";
      weight_whole.Resize({weight_numel});
      ctx.template Alloc<T>(&weight_whole);
      weight_to_tensor<T>(place, stream, running_weight_list, &weight_whole);
      w_data = weight_whole.data<T>();
      if (is_test) {  // maybe also reset small weights' ptr for training
        int offset = 0;
        for (size_t i = 0; i < running_weight_list.size(); ++i) {
          size_t len = running_weight_list[i]->numel();
          auto dim = running_weight_list[i]->dims();
          const_cast<phi::DenseTensor *>(running_weight_list[i])
              ->ShareDataWith(
                  weight_whole.Slice(static_cast<int64_t>(offset),
                                     static_cast<int64_t>(offset + len)))
              .Resize(dim);
          offset += len;
        }
      }
    } else {
      w_data = const_cast<T *>(running_weight_list[0]->data<T>());
    }
  } else {
    w_data = const_cast<T *>(running_w->data<T>());
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
  workspace_data_.Resize({static_cast<int64_t>(workspace_size)});
  ctx.template Alloc<uint8_t>(&workspace_data_);

  reserve->Resize({static_cast<int64_t>(reserve_size)});
  auto *reserve_data = ctx.template Alloc<uint8_t>(reserve);

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
#if CUDNN_VERSION >= 90000
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnRNNForward(handle,
                                      rnn.rnn_desc(),
                                      CUDNN_FWD_MODE_TRAINING,
                                      nullptr,
                                      rnn.x_seq_desc(),
                                      x_data,
                                      rnn.y_seq_desc(),
                                      out_data,
                                      rnn.init_h_desc(),
                                      init_h_data,
                                      last_h_data,
                                      rnn.init_c_desc(),
                                      init_c_data,
                                      last_c_data,
                                      rnn.weights_size(),
                                      w_data,
                                      workspace_size,
                                      workspace_data_.data<uint8_t>(),
                                      reserve_size,
                                      reserve_data));
#else

    if (!has_seq_length) {
// for train
// This interface is used when the input/output is unpadded.
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::miopenRNNForwardTraining(
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
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnRNNForwardTraining(handle,
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
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnRNNForwardTrainingEx(
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
      PADDLE_THROW(common::errors::Unavailable(
          "The padded input is supported by "
          "cudnnRNNForwardTrainingEx, but it only works when "
          "the version of cudnn is larger than 7.2.1"));
#endif
    }
#endif  // end CUDNN_VERSION >= 90000
  }
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(cudnn_lstm, GPU, ALL_LAYOUT, phi::CudnnLSTMKernel, float) {
  kernel->InputAt(5).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::UINT8);
  kernel->OutputAt(4).SetDataType(phi::DataType::UINT8);
}
#else
PD_REGISTER_KERNEL(
    cudnn_lstm, GPU, ALL_LAYOUT, phi::CudnnLSTMKernel, float, double) {
  kernel->InputAt(5).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::UINT8);
  kernel->OutputAt(4).SetDataType(phi::DataType::UINT8);
}
#endif
