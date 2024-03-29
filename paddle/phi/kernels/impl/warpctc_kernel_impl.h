// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <vector>

#include "paddle/phi/backends/dynload/warpctc.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/lod_utils.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sequence_padding.h"
#include "paddle/phi/kernels/funcs/sequence_scale.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename Context, typename T>
class ComputeCtcLossFunctor {
 public:
  ctcStatus_t operator()(const T* const activations,
                         T* gradients,
                         const int* const flat_labels,
                         const int* const label_lengths,
                         const int* const input_lengths,
                         int alphabet_size,
                         int minibatch,
                         T* costs,
                         void* workspace,
                         ctcOptions options) {
    return CTC_STATUS_EXECUTION_FAILED;
  }
};

template <typename Context>
class ComputeCtcLossFunctor<Context, float> {
 public:
  ctcStatus_t operator()(const float* const activations,
                         float* gradients,
                         const int* const flat_labels,
                         const int* const label_lengths,
                         const int* const input_lengths,
                         int alphabet_size,
                         int minibatch,
                         float* costs,
                         void* workspace,
                         ctcOptions options) {
    return phi::dynload::compute_ctc_loss(activations,
                                          gradients,
                                          flat_labels,
                                          label_lengths,
                                          input_lengths,
                                          static_cast<int>(alphabet_size),
                                          static_cast<int>(minibatch),
                                          costs,
                                          workspace,
                                          options);
  }
};

template <typename Context>
class ComputeCtcLossFunctor<Context, double> {
 public:
  ctcStatus_t operator()(const double* const activations,
                         double* gradients,
                         const int* const flat_labels,
                         const int* const label_lengths,
                         const int* const input_lengths,
                         int alphabet_size,
                         int minibatch,
                         double* costs,
                         void* workspace,
                         ctcOptions options) {
    return phi::dynload::compute_ctc_loss_double(
        activations,
        gradients,
        flat_labels,
        label_lengths,
        input_lengths,
        static_cast<int>(alphabet_size),
        static_cast<int>(minibatch),
        costs,
        workspace,
        options);
  }
};

template <typename Context, typename T>
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
  void operator()(const Context& dev_ctx,
                  const T* input,
                  T* gradient,
                  const int* cpu_labels,
                  const int* cpu_label_lengths,
                  const int* cpu_input_lengths,
                  const size_t sequence_width,
                  const size_t num_sequences,
                  const size_t blank,
                  T* cpu_loss) {
    // Init warp-ctc options
    init(dev_ctx, blank);

    // Compute the required workspace size.
    // There is no memory allocated operations within warp-ctc.
    size_t workspace_bytes = 0;
    ctcStatus_t status = CTC_STATUS_UNKNOWN_ERROR;
    if (sizeof(T) == 4) {
      status =
          phi::dynload::get_workspace_size(cpu_label_lengths,
                                           cpu_input_lengths,
                                           static_cast<int>(sequence_width),
                                           static_cast<int>(num_sequences),
                                           options_,
                                           &workspace_bytes);
    } else {
      status = phi::dynload::get_workspace_size_double(
          cpu_label_lengths,
          cpu_input_lengths,
          static_cast<int>(sequence_width),
          static_cast<int>(num_sequences),
          options_,
          &workspace_bytes);
    }
    PADDLE_ENFORCE_EQ(
        CTC_STATUS_SUCCESS,
        status,
        errors::PreconditionNotMet(
            "warp-ctc [version %d] Error in get_workspace_size: %s",
            warpctc_version_,
            phi::dynload::ctcGetStatusString(status)));
    PADDLE_ENFORCE_GT(
        workspace_bytes,
        0UL,
        errors::InvalidArgument(
            "Bytes of workspace got by warp-ctc function, "
            "get_workspace_size() should be larger than 0, but received %d",
            workspace_bytes));

    size_t workspace_elements = workspace_bytes / sizeof(T) + 1UL;
    DenseTensor workspace = phi::Empty<T, Context>(
        dev_ctx, {static_cast<int64_t>(workspace_elements)});
    T* workspace_data = workspace.data<T>();
    phi::funcs::SetConstant<Context, T>()(
        dev_ctx, &workspace, static_cast<T>(0));

    // compute loss and gradient
    status =
        ComputeCtcLossFunctor<Context, T>()(input,
                                            gradient,
                                            cpu_labels,
                                            cpu_label_lengths,
                                            cpu_input_lengths,
                                            static_cast<int>(sequence_width),
                                            static_cast<int>(num_sequences),
                                            cpu_loss,
                                            workspace_data,
                                            options_);

    PADDLE_ENFORCE_EQ(
        CTC_STATUS_SUCCESS,
        status,
        errors::PreconditionNotMet(
            "warp-ctc [version %d] Error in get_workspace_size: %s",
            warpctc_version_,
            phi::dynload::ctcGetStatusString(status)));
  }

 protected:
  void init(const Context& dev_ctx, const size_t blank) {
    warpctc_version_ = phi::dynload::get_warpctc_version();

    if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      options_.loc = CTC_GPU;
      options_.stream =
          reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
#else
      PADDLE_THROW(
          errors::PreconditionNotMet("[warpctc init] GPU is not enabled."));
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

template <typename T, typename Context>
void WarpctcKernel(const Context& dev_ctx,
                   const DenseTensor& logits,
                   const DenseTensor& label,
                   const paddle::optional<DenseTensor>& logits_length,
                   const paddle::optional<DenseTensor>& labels_length,
                   int blank,
                   bool norm_by_times UNUSED,
                   DenseTensor* loss,
                   DenseTensor* warpctcgrad) {
  size_t num_sequences, sequence_width, max_sequence_length;
  phi::Vector<size_t> logits_lod;
  phi::Vector<size_t> label_lod;
  if (logits_length.is_initialized() && labels_length.is_initialized()) {
    num_sequences = logits.dims()[1];
    sequence_width = logits.dims()[2];
    max_sequence_length = logits.dims()[0];

    PADDLE_ENFORCE_GT(max_sequence_length,
                      0,
                      phi::errors::InvalidArgument(
                          "The first dimension of Input(Logits) should be "
                          "greater than zero "
                          "but received %d. ",
                          max_sequence_length));

    PADDLE_ENFORCE_GT(num_sequences,
                      0,
                      phi::errors::InvalidArgument(
                          "The second dimension of Input(Logits) should be "
                          "greater than zero "
                          "but received %d. ",
                          num_sequences));

    PADDLE_ENFORCE_GT(sequence_width,
                      0,
                      phi::errors::InvalidArgument(
                          "The third dimension of Input(Logits) should be "
                          "greater than zero "
                          "but received %d. ",
                          sequence_width));

    DenseTensor logits_length_cpu;
    DenseTensor labels_length_cpu;
    phi::Copy(
        dev_ctx, *logits_length, phi::CPUPlace(), false, &logits_length_cpu);
    phi::Copy(
        dev_ctx, *labels_length, phi::CPUPlace(), false, &labels_length_cpu);

    logits_lod.push_back(0);
    label_lod.push_back(0);
    for (size_t i = 0; i < num_sequences; i++) {
      logits_lod.push_back(logits_lod[i] +
                           logits_length_cpu.data<int64_t>()[i]);
      label_lod.push_back(label_lod[i] + labels_length_cpu.data<int64_t>()[i]);
    }
  } else {
    PADDLE_ENFORCE_GT(
        logits.NumLevels(),
        0UL,
        phi::errors::InvalidArgument("Input(Logits) Tensor of WarpCTC "
                                     "does not contain LoD information."));
    PADDLE_ENFORCE_GT(
        label.NumLevels(),
        0UL,
        phi::errors::InvalidArgument("Input(Label) Tensor of WarpCTC "
                                     "does not contain LoD information."));

    logits_lod = phi::ToAbsOffset(logits.lod())[0];
    auto logits_dims = logits.dims();

    PADDLE_ENFORCE_GT(logits_dims[0],
                      0,
                      phi::errors::InvalidArgument(
                          "The first dimension of Input(Logits) should be "
                          "greater than zero "
                          "but received %d. ",
                          logits_dims[0]));

    PADDLE_ENFORCE_EQ(
        logits_dims[0],
        static_cast<int64_t>(logits_lod.back()),
        phi::errors::InvalidArgument(
            "The first dimension of Input(Logits) should be equal to "
            "the sum of all sequences' lengths = %d., but received %d. ",
            static_cast<int64_t>(logits_lod.back()),
            logits_dims[0]));

    label_lod = phi::ToAbsOffset(label.lod())[0];
    auto label_dims = label.dims();
    PADDLE_ENFORCE_EQ(label_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dimension of Input(Label) should be 1, "
                          "but received %d",
                          label_dims[1]));

    num_sequences = logits_lod.size() - 1;
    PADDLE_ENFORCE_EQ(num_sequences,
                      label_lod.size() - 1,
                      phi::errors::InvalidArgument(
                          "The number of sequences of Input(Logits) should be "
                          "equal to that of Input(Label) = %d, but received %d",
                          label_lod.size() - 1,
                          num_sequences));

    sequence_width = logits.numel() / logits_dims[0];
    max_sequence_length = phi::funcs::MaximumSequenceLength(logits_lod);
  }

  auto loss_dims = common::make_ddim({static_cast<int64_t>(num_sequences), 1});

  // warpctc needs sequences data stored in transposed padding format
  DenseTensor warpctc_logits_tmp =
      phi::Empty<T, Context>(dev_ctx,
                             {static_cast<int64_t>(max_sequence_length),
                              static_cast<int64_t>(num_sequences),
                              static_cast<int64_t>(sequence_width)});
  DenseTensor warpctc_logits(warpctc_logits_tmp);

  if (logits_length.is_initialized()) {
    phi::Copy(dev_ctx, logits, dev_ctx.GetPlace(), true, &warpctc_logits);
  } else {
    DenseTensor cpu_pad_value;
    cpu_pad_value.Resize({1});
    T* pad_value_data = dev_ctx.template HostAlloc<T>(&cpu_pad_value);
    *pad_value_data = static_cast<T>(0);
    DenseTensor pad_value;
    if (dev_ctx.GetPlace() == phi::CPUPlace()) {
      pad_value = cpu_pad_value;
    } else {
      phi::Copy(dev_ctx, cpu_pad_value, dev_ctx.GetPlace(), true, &pad_value);
    }

    phi::funcs::PaddingLoDTensorFunctor<Context, T>()(
        dev_ctx,
        logits,
        &warpctc_logits,
        pad_value,
        -1,
        0,
        false /* norm_by_times */,
        phi::funcs::kLengthBatchWidth);
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
  warpctcgrad->Resize(warpctc_logits.dims());
  T* warpctcgrad_data = dev_ctx.template Alloc<T>(warpctcgrad);

  phi::funcs::SetConstant<Context, T>()(
      dev_ctx, warpctcgrad, static_cast<T>(0));

  // warpctc accesses labels in CPU memory
  DenseTensor warpctc_label;
  if (logits_length.is_initialized()) {
    warpctc_label.Resize(
        {static_cast<int64_t>(phi::funcs::TotalSequenceLength(label_lod)), 1});
    dev_ctx.template HostAlloc<int>(&warpctc_label);
    std::vector<phi::Vector<size_t>> lod;
    lod.push_back(label_lod);
    warpctc_label.set_lod(lod);

    if (dev_ctx.GetPlace() == phi::CPUPlace()) {
      phi::funcs::UnpaddingLoDTensorFunctor<Context, int>()(
          dev_ctx,
          label,
          &warpctc_label,
          label.dims()[1] /*pad_seq_len*/,
          0 /*lod_level*/,
          false /*norm_by_times*/,
          phi::funcs::kBatchLengthWidth);
    } else {
      DenseTensor gpu_label;
      gpu_label.Resize(
          {static_cast<int64_t>(phi::funcs::TotalSequenceLength(label_lod)),
           1});
      dev_ctx.template Alloc<int>(&gpu_label);
      gpu_label.set_lod(lod);
      phi::funcs::UnpaddingLoDTensorFunctor<Context, int>()(
          dev_ctx,
          label,
          &gpu_label,
          label.dims()[1] /*pad_seq_len*/,
          0 /*lod_level*/,
          false /*norm_by_times*/,
          phi::funcs::kBatchLengthWidth);
      phi::Copy(dev_ctx, gpu_label, phi::CPUPlace(), true, &warpctc_label);
    }
  } else {
    phi::Copy(dev_ctx, label, phi::CPUPlace(), true, &warpctc_label);
  }

  const int* warpctc_label_data = warpctc_label.data<int>();
  // warpctc stores loss in CPU memory
  DenseTensor warpctc_loss;
  warpctc_loss.Resize(loss_dims);
  T* warpctc_loss_data = dev_ctx.template HostAlloc<T>(&warpctc_loss);
  WarpCTCFunctor<Context, T>()(dev_ctx,
                               warpctc_logits_data,
                               warpctcgrad_data,
                               warpctc_label_data,
                               warpctc_label_lengths.data(),
                               warpctc_logits_lengths.data(),
                               sequence_width,
                               num_sequences,
                               blank,
                               warpctc_loss_data);
  // Copy the loss back
  phi::Copy(dev_ctx, warpctc_loss, dev_ctx.GetPlace(), false, loss);
}

}  // namespace phi
