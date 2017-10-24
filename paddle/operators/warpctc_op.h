/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/seq2batch.h"
#include "paddle/platform/dynload/warpctc.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

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
    PADDLE_ENFORCE_EQ(CTC_STATUS_SUCCESS, status,
                      "warp-ctc [version %d] Error in get_workspace_size: ",
                      warpctc_version_,
                      platform::dynload::ctcGetStatusString(status));
    PADDLE_ENFORCE_GT(workspace_bytes, 0UL,
                      "Bytes of workspace got by warp-ctc function, "
                      "get_workspace_size(), should be larger than 0.");

    Tensor workspace;
    size_t workspace_elements = workspace_bytes / sizeof(float) + 1UL;
    float* workspace_data = workspace.mutable_data<float>(
        framework::make_ddim({static_cast<int64_t>(workspace_elements)}),
        ctx.GetPlace());

    // compute loss and gradient
    status = platform::dynload::compute_ctc_loss(
        input, gradient, cpu_labels, cpu_label_lengths, cpu_input_lengths,
        static_cast<int>(sequence_width), static_cast<int>(num_sequences),
        cpu_loss, workspace_data, options_);
    PADDLE_ENFORCE_EQ(CTC_STATUS_SUCCESS, status,
                      "warp-ctc [version %d] Error in compute_ctc_loss: ",
                      warpctc_version_,
                      platform::dynload::ctcGetStatusString(status));
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

template <typename Place, typename T>
class WarpCTCKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* logits = ctx.Input<LoDTensor>("Logits");
    auto* label = ctx.Input<LoDTensor>("Label");
    auto* warpctc_grad = ctx.Output<Tensor>("WarpCTCGrad");
    auto* loss = ctx.Output<Tensor>("Loss");

    const size_t level = 0;

    auto logits_lod = logits->lod();
    auto logits_dims = logits->dims();
    PADDLE_ENFORCE_EQ(logits_dims[0],
                      static_cast<int64_t>(logits_lod[level].back()),
                      "The first dimension of Input(Logits) should be equal to "
                      "the sum of all sequences' lengths.");

    auto label_lod = label->lod();
    auto label_dims = label->dims();
    PADDLE_ENFORCE_EQ(
        label_dims[0], label->numel(),
        "The width of each timestep in Input(Label) should be 1.");

    const size_t num_sequences = logits_lod[level].size() - 1;
    PADDLE_ENFORCE_EQ(num_sequences, label_lod[level].size() - 1,
                      "The number of sequences of Input(Logits) should be "
                      "equal to that of Input(Label).");

    const size_t sequence_width = logits->numel() / logits_dims[0];
    auto loss_dims =
        framework::make_ddim({static_cast<int64_t>(num_sequences), 1});

    // warpctc needs sequences data stored in batch format
    Tensor warpctc_logits;
    math::Seq2BatchFunctor<true, Place, T>()(ctx.device_context(), *logits,
                                             warpctc_logits, false);
    const T* warpctc_logits_data = warpctc_logits.data<T>();

    std::vector<int> warpctc_label_lengths(num_sequences);
    std::vector<int> warpctc_logits_lengths(num_sequences);

    for (size_t i = 0; i < num_sequences; ++i) {
      warpctc_label_lengths[i] = label_lod[level][i + 1] - label_lod[level][i];
      warpctc_logits_lengths[i] =
          logits_lod[level][i + 1] - logits_lod[level][i];
    }

    // warpctc computes loss and gradient in one call, gradient data also stored
    // in batch format
    T* warpctc_grad_data =
        warpctc_grad->mutable_data<T>(warpctc_logits.dims(), ctx.GetPlace());

    // warpctc accesses labels in CPU memory
    Tensor warpctc_label;
    warpctc_label.CopyFrom(*label, platform::CPUPlace(), ctx.device_context());
    const int* warpctc_label_data = warpctc_label.data<int>();

    // warpctc stores loss in CPU memory
    Tensor warpctc_loss;
    T* warpctc_loss_data =
        warpctc_loss.mutable_data<T>(loss_dims, platform::CPUPlace());

    const size_t blank = static_cast<size_t>(ctx.Attr<int>("blank"));

    WarpCTCFunctor()(ctx, warpctc_logits_data, warpctc_grad_data,
                     warpctc_label_data, warpctc_label_lengths.data(),
                     warpctc_logits_lengths.data(), sequence_width,
                     num_sequences, blank, warpctc_loss_data);
    std::cout << "Loss in warpctc_op:" << std::endl;
    for (size_t i = 0; i < num_sequences; i++)
      std::cout << warpctc_loss_data[i] << std::endl;

    // Copy the loss back
    loss->CopyFrom(warpctc_loss, ctx.GetPlace(), ctx.device_context());
  }
};

template <typename Place, typename T>
class WarpCTCGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* warpctc_grad = ctx.Input<Tensor>("WarpCTCGrad");
    auto* logits_grad = ctx.Output<LoDTensor>(framework::GradVarName("Logits"));

    bool norm_by_times = ctx.Attr<bool>("normByTimes");

    math::Batch2SeqFunctor<true, Place, T>()(ctx.device_context(), *logits_grad,
                                             *warpctc_grad, norm_by_times);
  }
};

}  // namespace operators
}  // namespace paddle
