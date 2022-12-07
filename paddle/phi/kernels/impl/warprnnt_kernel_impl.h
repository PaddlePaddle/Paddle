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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/warp_transducer/include/rnnt.h"

namespace phi {

template <typename Context, typename T>
class ComputeRnntLossFunctor {
 public:
  rnntStatus_t operator()(const T* const activations,
                          T* gradients,
                          const int* const flat_labels,
                          const int* const label_lengths,
                          const int* const input_lengths,
                          int alphabet_size,
                          int minibatch,
                          T* costs,
                          void* workspace,
                          rnntOptions options) {
    return RNNT_STATUS_EXECUTION_FAILED;
  }
};

template <typename Context>
class ComputeRnntLossFunctor<Context, float> {
 public:
  rnntStatus_t operator()(const float* const activations,
                          float* gradients,
                          const int* const flat_labels,
                          const int* const label_lengths,
                          const int* const input_lengths,
                          int alphabet_size,
                          int minibatch,
                          float* costs,
                          void* workspace,
                          rnntOptions options) {
    std::cout << "float32..." << std::endl;
    return compute_rnnt_loss(activations,
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
class ComputeRnntLossFunctor<Context, double> {
 public:
  rnntStatus_t operator()(const double* const activations,
                          double* gradients,
                          const int* const flat_labels,
                          const int* const label_lengths,
                          const int* const input_lengths,
                          int alphabet_size,
                          int minibatch,
                          double* costs,
                          void* workspace,
                          rnntOptions options) {
    std::cout << "float64..." << std::endl;
    return compute_rnnt_loss_fp64(activations,
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
class WarpRNNTFunctor {
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
   *                          B x T x U x D, (row-major) format
   * \param gradient          batch matrix of gradient, with the same shape as
   *                          input.
   * \param cpu_labels        labels always in CPU memory.
   * \param cpu_label_lengths length of all labels in CPU memory.
   * \param cpu_input_lengths length of all sequences in CPU memory.
   * \param D    number of possible output symbols.
   * \param B     number of sequence.
   * \param blank             blank label used in ctc loss function.
   * \param cpu_losss         cost of each sequence in CPU memory.
   */
  void operator()(const Context& dev_ctx,
                  const T* input,
                  T* gradient,
                  const int* cpu_labels,
                  const int* cpu_label_lengths,
                  const int* cpu_input_lengths,
                  const size_t D,
                  const size_t B,
                  const size_t maxT,
                  const size_t maxU,
                  const int blank,
                  const float fastemit_lambda,
                  const int num_threads,
                  T* cpu_loss) {
    // Init warp-rnnt options
    init(dev_ctx, maxT, maxU, blank, fastemit_lambda, num_threads);

    // Compute the required workspace size.
    // There is no memory allocated operations within warp-rnnt.
    rnntStatus_t status = RNNT_STATUS_UNKNOWN_ERROR;
    bool gpu = false;
    if (paddle::platform::is_gpu_place(dev_ctx.GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      gpu = true;
#else
      PADDLE_THROW(errors::PreconditionNotMet(
          "[WarpRNNTFunctor Operator] GPU is not enabled."));
#endif
    }

    size_t workspace_bytes = 0;
    status = get_rnnt_workspace_size(
        maxT, maxU, B, gpu, &workspace_bytes, sizeof(T));
    std::cout << "B: " << B << std::endl;
    std::cout << "maxT: " << maxT << std::endl;
    std::cout << "maxU: " << maxU << std::endl;
    std::cout << "D: " << D << std::endl;
    std::cout << "gpu: " << gpu << std::endl;
    std::cout << "worspace_bytes: " << workspace_bytes << std::endl;

    PADDLE_ENFORCE_EQ(
        RNNT_STATUS_SUCCESS,
        status,
        errors::PreconditionNotMet(
            "warp-rnnt [version %d] Error in get_workspace_size: %s",
            warprnnt_version_,
            rnntGetStatusString(status)));
    PADDLE_ENFORCE_GT(
        workspace_bytes,
        0UL,
        errors::InvalidArgument(
            "Bytes of workspace got by warp-rnnt function, "
            "get_workspace_size() should be larger than 0, but received %d",
            workspace_bytes));

    size_t workspace_elements = workspace_bytes / sizeof(T) + 1UL;
    DenseTensor workspace = phi::Full<T, Context>(
        dev_ctx, {static_cast<int64_t>(workspace_elements)}, static_cast<T>(0));
    T* workspace_data = workspace.data<T>();
    std::cout << "set workspace: " << workspace << std::endl;
    // compute loss and gradient
    status = ComputeRnntLossFunctor<Context, T>()(input,
                                                  gradient,
                                                  cpu_labels,
                                                  cpu_label_lengths,
                                                  cpu_input_lengths,
                                                  static_cast<int>(D),
                                                  static_cast<int>(B),
                                                  cpu_loss,
                                                  workspace_data,
                                                  options_);
    std::cout << "ComputeRnntLossFunctor done" << std::endl;
    PADDLE_ENFORCE_EQ(
        RNNT_STATUS_SUCCESS,
        status,
        errors::PreconditionNotMet(
            "warp-rnnt [version %d] Error in get_workspace_size: %s",
            warprnnt_version_,
            rnntGetStatusString(status)));
  }

 protected:
  void init(const Context& dev_ctx,
            const size_t maxT,
            const size_t maxU,
            const size_t blank,
            const float fastemit_lambda,
            const int num_threads) {
    warprnnt_version_ = get_warprnnt_version();

    options_.maxT = maxT;
    options_.maxU = maxU;
    options_.blank_label = blank;
    options_.fastemit_lambda = fastemit_lambda;
    options_.batch_first = true;

    if (paddle::platform::is_gpu_place(dev_ctx.GetPlace())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      options_.loc = RNNT_GPU;
      options_.stream =
          reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
#else
      PADDLE_THROW(
          errors::PreconditionNotMet("[warprnnt init] GPU is not enabled."));
#endif
    } else {
      options_.loc = RNNT_CPU;
      options_.num_threads = num_threads;
#ifdef PADDLE_WITH_MKLML
      // have to use at least one
      options_.num_threads = std::max(options_.num_threads, (unsigned int)1);
#endif
    }
  }

 private:
  int warprnnt_version_;
  rnntOptions options_;
};

template <typename T, typename Context>
void WarprnntKernel(const Context& dev_ctx,
                    const DenseTensor& logits,
                    const DenseTensor& label,
                    const DenseTensor& logits_length,
                    const DenseTensor& labels_length,
                    int blank,
                    float fastemit_lambda,
                    int num_threads,
                    DenseTensor* loss,
                    DenseTensor* warprnntgrad) {
  PADDLE_ENFORCE_EQ(
      logits.dims().size(),
      4,
      phi::errors::InvalidArgument("The rank of Input(Logits) should be 4 "
                                   "but received %d. ",
                                   logits.dims().size()));

  PADDLE_ENFORCE_EQ(
      label.dims().size(),
      2,
      phi::errors::InvalidArgument("The rank of Input(Label) should be 2 "
                                   "but received %d. ",
                                   label.dims().size()));

  PADDLE_ENFORCE_EQ(logits_length.dims().size(),
                    1,
                    phi::errors::InvalidArgument(
                        "The rank of Input(LogitsLength) should be 1 "
                        "but received %d. ",
                        logits_length.dims().size()));

  PADDLE_ENFORCE_EQ(
      labels_length.dims().size(),
      1,
      phi::errors::InvalidArgument("The rank of Input(LabelLength) should be 1 "
                                   "but received %d. ",
                                   labels_length.dims().size()));

  size_t B, Tmax, Umax, D;
  B = logits.dims()[0];
  Tmax = logits.dims()[1];
  Umax = logits.dims()[2];
  D = logits.dims()[3];
  std::cout << "input shape: " << B << "," << Tmax << "," << Umax << "," << D
            << std::endl;

  PADDLE_ENFORCE_GT(B,
                    0,
                    phi::errors::InvalidArgument(
                        "The first dimension of Input(Logits) is B should be "
                        "greater than zero "
                        "but received %d. ",
                        B));

  PADDLE_ENFORCE_GT(Tmax,
                    0,
                    phi::errors::InvalidArgument(
                        "The second dimension of Input(Logits) is T should be "
                        "greater than zero "
                        "but received %d. ",
                        Tmax));

  PADDLE_ENFORCE_GT(Umax,
                    0,
                    phi::errors::InvalidArgument(
                        "The third dimension of Input(Logits) is U should be "
                        "greater than zero "
                        "but received %d. ",
                        Umax));

  PADDLE_ENFORCE_GT(D,
                    0,
                    phi::errors::InvalidArgument(
                        "The forth dimension of Input(Logits) is D should be "
                        "greater than zero "
                        "but received %d. ",
                        D));
  std::cout << "logits: " << logits << std::endl;
  std::cout << "labels: " << label << std::endl;
  std::cout << "logits_length: " << logits_length << std::endl;
  std::cout << "labels_length: " << labels_length << std::endl;

  warprnntgrad->Resize(logits.dims());
  T* warprnntgrad_data = dev_ctx.template Alloc<T>(warprnntgrad);
  phi::funcs::SetConstant<Context, T>()(
      dev_ctx, warprnntgrad, static_cast<T>(0));

  // Loss (B)
  auto loss_dims = phi::make_ddim({static_cast<int64_t>(B)});
  DenseTensor warprnnt_loss;
  warprnnt_loss.Resize(loss_dims);
  T* warprnnt_loss_data = dev_ctx.template HostAlloc<T>(&warprnnt_loss);
  std::cout << "warprnnt_loss: " << warprnnt_loss << std::endl;

  WarpRNNTFunctor<Context, T>()(dev_ctx,
                                logits.data<T>(),
                                warprnntgrad_data,
                                label.data<int>(),
                                labels_length.data<int>(),
                                logits_length.data<int>(),
                                D,
                                B,
                                Tmax,
                                Umax,
                                blank,
                                fastemit_lambda,
                                num_threads,
                                warprnnt_loss_data);
  std::cout << "warprnntgrad: " << *warprnntgrad << std::endl;

  phi::Copy(dev_ctx, warprnnt_loss, dev_ctx.GetPlace(), true, loss);
  std::cout << "rnnt kernel done." << std::endl;
}

}  // namespace phi
