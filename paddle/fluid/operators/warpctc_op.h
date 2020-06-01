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

#pragma once

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence_padding.h"
#include "paddle/fluid/operators/math/sequence_scale.h"
#include "paddle/fluid/platform/dynload/warpctc.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext>
class WarpCTCFunctor {
 public:
  /*
   * \brief Compute the connectionist temporal classification loss,
   *        and optionally compute the gradient with respect to the inputs.
   *
   * If gradient is nullptr, it only computes the ctc loss,
   * or computes both ctc loss and gradient.
   *
   * \param ctx               execution context of this functor
   * \param input             batch matrix of input probabilities, in
   *                          max_sequence_length x num_sequences x
   *                          sequence_width, (row-major) format
   * \param gradient          batch matrix of gradient, with the same shape as
   *                          input.
   * \param cpu_labels        labels always in CPU memory.
   * \param cpu_label_lengths length of all labels in CPU memory.
   * \param cpu_input_lengths length of all sequences in CPU memory.
   * \param sequence_width    number of possible output symbols.
   * \param num_sequences     number of sequence.
   * \param blank             blank label used in ctc loss function.
   * \param cpu_losss         cost of each sequence in CPU memory.
   */
  void operator()(const framework::ExecutionContext& ctx, const float* input,
                  float* gradient, const int* cpu_labels,
                  const int* cpu_label_lengths, const int* cpu_input_lengths,
                  const size_t sequence_width, const size_t num_sequences,
                  const size_t blank, float* cpu_loss) {
    // Init warp-ctc options
    init(ctx, blank);

    // Compute the required workspace size.
    // There is no memory allocated operations within warp-ctc.
    size_t workspace_bytes = 0;
    ctcStatus_t status = platform::dynload::get_workspace_size(
        cpu_label_lengths, cpu_input_lengths, static_cast<int>(sequence_width),
        static_cast<int>(num_sequences), options_, &workspace_bytes);

    PADDLE_ENFORCE_EQ(
        CTC_STATUS_SUCCESS, status,
        platform::errors::PreconditionNotMet(
            "warp-ctc [version %d] Error in get_workspace_size: %s",
            warpctc_version_, platform::dynload::ctcGetStatusString(status)));
    PADDLE_ENFORCE_GT(
        workspace_bytes, 0UL,
        platform::errors::InvalidArgument(
            "Bytes of workspace got by warp-ctc function, "
            "get_workspace_size() should be larger than 0, but received %d",
            workspace_bytes));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    size_t workspace_elements = workspace_bytes / sizeof(float) + 1UL;
    Tensor workspace = ctx.AllocateTmpTensor<float, DeviceContext>(
        framework::make_ddim({static_cast<int64_t>(workspace_elements)}),
        dev_ctx);
    float* workspace_data = workspace.data<float>();
    math::SetConstant<DeviceContext, float>()(
        ctx.template device_context<DeviceContext>(), &workspace,
        static_cast<float>(0));

    // compute loss and gradient
    status = platform::dynload::compute_ctc_loss(
        input, gradient, cpu_labels, cpu_label_lengths, cpu_input_lengths,
        static_cast<int>(sequence_width), static_cast<int>(num_sequences),
        cpu_loss, workspace_data, options_);

    PADDLE_ENFORCE_EQ(
        CTC_STATUS_SUCCESS, status,
        platform::errors::PreconditionNotMet(
            "warp-ctc [version %d] Error in get_workspace_size: %s",
            warpctc_version_, platform::dynload::ctcGetStatusString(status)));
  }

 protected:
  void init(const framework::ExecutionContext& ctx, const size_t blank) {
    warpctc_version_ = platform::dynload::get_warpctc_version();

    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
      options_.loc = CTC_GPU;
      options_.stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                            ctx.device_context())
                            .stream();
#else
      PADDLE_THROW("[warpctc init] GPU is not enabled.");
#endif
    } else {
      options_.loc = CTC_CPU;
      options_.num_threads = 1;
    }

    options_.blank_label = blank;
  }

 private:
  int warpctc_version_;
  ctcOptions options_;
};

template <typename DeviceContext, typename T>
class WarpCTCKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* logits = ctx.Input<LoDTensor>("Logits");
    auto* label = ctx.Input<LoDTensor>("Label");
    auto* warpctc_grad = ctx.Output<Tensor>("WarpCTCGrad");
    auto* loss = ctx.Output<Tensor>("Loss");

    size_t num_sequences, sequence_width, max_sequence_length;
    framework::Vector<size_t> logits_lod;
    framework::Vector<size_t> label_lod;

    if (ctx.HasInput("LogitsLength") && ctx.HasInput("LabelLength")) {
      num_sequences = logits->dims()[1];
      sequence_width = logits->dims()[2];
      max_sequence_length = logits->dims()[0];

      auto* logits_length = ctx.Input<framework::Tensor>("LogitsLength");
      auto* labels_length = ctx.Input<framework::Tensor>("LabelLength");
      framework::Tensor logits_length_cpu;
      framework::Tensor labels_length_cpu;
      framework::TensorCopy(*logits_length, platform::CPUPlace(),
                            &logits_length_cpu);
      framework::TensorCopy(*labels_length, platform::CPUPlace(),
                            &labels_length_cpu);

      logits_lod.push_back(0);
      label_lod.push_back(0);
      for (size_t i = 0; i < num_sequences; i++) {
        logits_lod.push_back(logits_lod[i] +
                             logits_length_cpu.data<int64_t>()[i]);
        label_lod.push_back(label_lod[i] +
                            labels_length_cpu.data<int64_t>()[i]);
      }
    } else {
      PADDLE_ENFORCE_GT(logits->NumLevels(), 0UL,
                        platform::errors::InvalidArgument(
                            "Input(Logits) Tensor of WarpCTC "
                            "does not contain LoD information."));
      PADDLE_ENFORCE_GT(label->NumLevels(), 0UL,
                        platform::errors::InvalidArgument(
                            "Input(Label) Tensor of WarpCTC "
                            "does not contain LoD information."));

      logits_lod = framework::ToAbsOffset(logits->lod())[0];
      auto logits_dims = logits->dims();

      PADDLE_ENFORCE_EQ(
          logits_dims[0], static_cast<int64_t>(logits_lod.back()),
          platform::errors::InvalidArgument(
              "The first dimension of Input(Logits) should be equal to "
              "the sum of all sequences' lengths = %d., but received %d. ",
              static_cast<int64_t>(logits_lod.back()), logits_dims[0]));

      label_lod = framework::ToAbsOffset(label->lod())[0];
      auto label_dims = label->dims();
      PADDLE_ENFORCE_EQ(label_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "The last dimension of Input(Label) should be 1, "
                            "but received %d",
                            label_dims[1]));

      num_sequences = logits_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          num_sequences, label_lod.size() - 1,
          platform::errors::InvalidArgument(
              "The number of sequences of Input(Logits) should be "
              "equal to that of Input(Label) = %d, but received %d",
              label_lod.size() - 1, num_sequences));

      sequence_width = logits->numel() / logits_dims[0];
      max_sequence_length = math::MaximumSequenceLength(logits_lod);
    }

    auto loss_dims =
        framework::make_ddim({static_cast<int64_t>(num_sequences), 1});

    // warpctc needs sequences data stored in transposed padding format
    LoDTensor warpctc_logits;
    auto warpctc_logits_dims =
        framework::make_ddim({static_cast<int64_t>(max_sequence_length),
                              static_cast<int64_t>(num_sequences),
                              static_cast<int64_t>(sequence_width)});
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    Tensor warpctc_logits_tmp =
        ctx.AllocateTmpTensor<T, DeviceContext>(warpctc_logits_dims, dev_ctx);
    warpctc_logits.ShareDataWith(warpctc_logits_tmp);
    if (ctx.HasInput("LogitsLength")) {
      TensorCopySync(*logits, ctx.GetPlace(), &warpctc_logits);
    } else {
      LoDTensor cpu_pad_value;
      T* pad_value_data =
          cpu_pad_value.mutable_data<T>({1}, platform::CPUPlace());
      *pad_value_data = static_cast<T>(0);
      LoDTensor pad_value;
      if (platform::is_cpu_place(ctx.GetPlace())) {
        pad_value = cpu_pad_value;
      } else {
        TensorCopySync(cpu_pad_value, ctx.GetPlace(), &pad_value);
      }

      math::PaddingLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), *logits,
          &warpctc_logits, pad_value, -1, 0, false /* norm_by_times */,
          math::kLengthBatchWidth);
    }
    const T* warpctc_logits_data = warpctc_logits.data<T>();

    std::vector<int> warpctc_label_lengths(num_sequences);
    std::vector<int> warpctc_logits_lengths(num_sequences);

    for (size_t i = 0; i < num_sequences; ++i) {
      warpctc_label_lengths[i] = label_lod[i + 1] - label_lod[i];
      warpctc_logits_lengths[i] = logits_lod[i + 1] - logits_lod[i];
    }

    // warpctc computes loss and gradient in one call, gradient data also stored
    // in batch format
    T* warpctc_grad_data =
        warpctc_grad->mutable_data<T>(warpctc_logits.dims(), ctx.GetPlace());

    math::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), warpctc_grad,
        static_cast<T>(0));

    // warpctc accesses labels in CPU memory
    LoDTensor warpctc_label;
    if (ctx.HasInput("LogitsLength")) {
      warpctc_label.mutable_data<int>(
          {static_cast<int64_t>(math::TotalSequenceLength(label_lod)), 1},
          platform::CPUPlace());
      std::vector<framework::Vector<size_t>> lod;
      lod.push_back(label_lod);
      warpctc_label.set_lod(lod);

      if (platform::is_cpu_place(ctx.GetPlace())) {
        math::UnpaddingLoDTensorFunctor<DeviceContext, int>()(
            ctx.template device_context<DeviceContext>(), *label,
            &warpctc_label, label->dims()[1] /*pad_seq_len*/, 0 /*lod_level*/,
            false /*norm_by_times*/, math::kBatchLengthWidth);
      } else {
        LoDTensor gpu_label;
        gpu_label.mutable_data<int>(
            {static_cast<int64_t>(math::TotalSequenceLength(label_lod)), 1},
            ctx.GetPlace());
        gpu_label.set_lod(lod);
        math::UnpaddingLoDTensorFunctor<DeviceContext, int>()(
            ctx.template device_context<DeviceContext>(), *label, &gpu_label,
            label->dims()[1] /*pad_seq_len*/, 0 /*lod_level*/,
            false /*norm_by_times*/, math::kBatchLengthWidth);
        TensorCopySync(gpu_label, platform::CPUPlace(), &warpctc_label);
      }
    } else {
      TensorCopySync(*label, platform::CPUPlace(), &warpctc_label);
    }

    const int* warpctc_label_data = warpctc_label.data<int>();
    // warpctc stores loss in CPU memory
    Tensor warpctc_loss;
    T* warpctc_loss_data =
        warpctc_loss.mutable_data<T>(loss_dims, platform::CPUPlace());

    const size_t blank = static_cast<size_t>(ctx.Attr<int>("blank"));

    WarpCTCFunctor<DeviceContext>()(
        ctx, warpctc_logits_data, warpctc_grad_data, warpctc_label_data,
        warpctc_label_lengths.data(), warpctc_logits_lengths.data(),
        sequence_width, num_sequences, blank, warpctc_loss_data);

    // Copy the loss back
    TensorCopy(warpctc_loss, ctx.GetPlace(), ctx.device_context(), loss);
  }
};

template <typename DeviceContext, typename T>
class WarpCTCGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* warpctc_grad = ctx.Input<LoDTensor>("WarpCTCGrad");
    auto* logits_grad = ctx.Output<LoDTensor>(framework::GradVarName("Logits"));
    const Tensor* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));

    logits_grad->mutable_data<T>(ctx.GetPlace());
    bool norm_by_times = ctx.Attr<bool>("norm_by_times");

    if (ctx.HasInput("LogitsLength")) {
      size_t max_seq_length = warpctc_grad->dims()[0];
      size_t num_sequences = warpctc_grad->dims()[1];
      size_t seq_width = warpctc_grad->dims()[2];

      auto* logits_length = ctx.Input<framework::Tensor>("LogitsLength");
      framework::Tensor logits_length_cpu;
      framework::TensorCopy(*logits_length, platform::CPUPlace(),
                            &logits_length_cpu);

      LoDTensor logits_grad_with_lod;
      auto logits_grad_dims =
          framework::make_ddim({static_cast<int64_t>(max_seq_length),
                                static_cast<int64_t>(num_sequences),
                                static_cast<int64_t>(seq_width)});
      T* logits_grad_cpu_data = logits_grad_with_lod.mutable_data<T>(
          logits_grad_dims, platform::CPUPlace());

      TensorCopySync(*warpctc_grad, platform::CPUPlace(),
                     &logits_grad_with_lod);

      Tensor loss_grad_cpu;
      loss_grad_cpu.mutable_data<T>(loss_grad->dims(), platform::CPUPlace());
      TensorCopySync(*loss_grad, platform::CPUPlace(), &loss_grad_cpu);

      LoDTensor scaled_logits;
      T* scaled_logits_data =
          scaled_logits.mutable_data<T>(logits_grad_dims, platform::CPUPlace());

      const T* loss_grad_data = loss_grad_cpu.data<T>();
      for (size_t i = 0; i < max_seq_length; ++i) {
        for (size_t j = 0; j < num_sequences; ++j) {
          T scale = 1.0;
          if (norm_by_times) {
            scale = 1.0 / static_cast<T>(logits_length_cpu.data<int64_t>()[j]);
          }
          for (size_t k = 0; k < seq_width; ++k) {
            size_t idx = i * (num_sequences * seq_width) + j * seq_width + k;
            scaled_logits_data[idx] =
                logits_grad_cpu_data[idx] * loss_grad_data[j] * scale;
          }
        }
      }

      TensorCopySync(scaled_logits, ctx.GetPlace(), logits_grad);
    } else {
      math::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), *warpctc_grad,
          logits_grad, -1, 0, norm_by_times, math::kLengthBatchWidth);

      const T* loss_grad_data = loss_grad->data<T>();
      math::ScaleLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), loss_grad_data,
          logits_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
