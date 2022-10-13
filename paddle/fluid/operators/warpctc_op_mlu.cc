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

#ifdef PADDLE_WITH_MLU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class WarpctcMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* logits = ctx.Input<Tensor>("Logits");
    auto* logits_lengths = ctx.Input<Tensor>("LogitsLength");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* labels_lengths = ctx.Input<Tensor>("LabelLength");
    auto* loss = ctx.Output<Tensor>("Loss");
    auto* grads = ctx.Output<Tensor>("WarpCTCGrad");
    const int blank = ctx.Attr<int>("blank");
    size_t num_sequences, sequence_width, max_sequence_length;
    paddle::framework::Vector<size_t> logits_lod;
    paddle::framework::Vector<size_t> label_lod;
    if (logits_lengths->initialized() && labels_lengths->initialized()) {
      num_sequences = logits->dims()[1];
      sequence_width = logits->dims()[2];
      max_sequence_length = logits->dims()[0];

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

    } else {
      PADDLE_ENFORCE_GT(
          logits->NumLevels(),
          0UL,
          phi::errors::InvalidArgument("Input(Logits) Tensor of WarpCTC "
                                       "does not contain LoD information."));
      PADDLE_ENFORCE_GT(
          labels->NumLevels(),
          0UL,
          phi::errors::InvalidArgument("Input(Label) Tensor of WarpCTC "
                                       "does not contain LoD information."));

      logits_lod = paddle::framework::ToAbsOffset(logits->lod())[0];
      auto logits_dims = logits->dims();

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

      label_lod = paddle::framework::ToAbsOffset(labels->lod())[0];
      auto label_dims = labels->dims();
      PADDLE_ENFORCE_EQ(label_dims[1],
                        1,
                        phi::errors::InvalidArgument(
                            "The last dimension of Input(Label) should be 1, "
                            "but received %d",
                            label_dims[1]));

      num_sequences = logits_lod.size() - 1;
      PADDLE_ENFORCE_EQ(
          num_sequences,
          label_lod.size() - 1,
          phi::errors::InvalidArgument(
              "The number of sequences of Input(Logits) should be "
              "equal to that of Input(Label) = %d, but received %d",
              label_lod.size() - 1,
              num_sequences));
    }
    // warpctc computes loss and gradient in one call, gradient data also stored
    // in batch format

    cnnlCTCLossNormalizationMode_t norm_mode;

    norm_mode = CNNL_NONE_NORMALIZATION;
    cnnlCTCLossReduceMode_t reduce_mode = CNNL_REDUCE_MODE_NONE;
    cnnlCTCLossZeroInfinityMode_t infinity_mode = CNNL_ZERO_INFINITY;
    int max_input_length = static_cast<int>(logits->dims()[0]);
    int temp_max_label_length = 0;
    std::vector<int> vec_labels_lengths;
    paddle::framework::TensorToVector(
        *labels_lengths, ctx.device_context(), &vec_labels_lengths);
    ctx.device_context().Wait();
    int labels_lengths_size = vec_labels_lengths.size();
    for (int i = 0; i < labels_lengths_size; ++i) {
      if (vec_labels_lengths[i] > temp_max_label_length)
        temp_max_label_length = vec_labels_lengths[i];
    }
    int max_label_length = static_cast<int>(temp_max_label_length);
    loss->Resize(phi::make_ddim({logits->dims()[1]}));
    loss->mutable_data<T>(ctx.GetPlace());
    grads->Resize(logits->dims());
    grads->mutable_data<T>(ctx.GetPlace());
    MLUCnnlTensorDesc in_desc(*logits, CNNL_LAYOUT_TNC, CNNL_DTYPE_FLOAT);
    MLUCnnlTensorDesc in_lengths_desc(*logits_lengths);
    MLUCnnlTensorDesc labels_desc(*labels);
    MLUCnnlTensorDesc labels_lengths_desc(*labels_lengths);
    MLUCnnlTensorDesc loss_desc(*loss);
    MLUCnnlTensorDesc grads_desc(*grads, CNNL_LAYOUT_TNC, CNNL_DTYPE_FLOAT);
    MLUCnnl::CTCLoss(ctx,
                     norm_mode,
                     reduce_mode,
                     infinity_mode,
                     blank,
                     max_input_length,
                     max_label_length,
                     in_desc.get(),
                     GetBasePtr(logits),
                     labels_desc.get(),
                     GetBasePtr(labels),
                     in_lengths_desc.get(),
                     GetBasePtr(logits_lengths),
                     labels_lengths_desc.get(),
                     GetBasePtr(labels_lengths),
                     loss_desc.get(),
                     GetBasePtr(loss),
                     grads_desc.get(),
                     GetBasePtr(grads));
  }
};

template <typename DeviceContext, typename T>
class WarpctcGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dlogits = ctx.Output<Tensor>(framework::GradVarName("Logits"));
    auto* grads = ctx.Input<Tensor>("WarpCTCGrad");
    dlogits->ShareDataWith(*grads);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    warpctc, ops::WarpctcMLUKernel<paddle::platform::MLUDeviceContext, float>);

REGISTER_OP_MLU_KERNEL(
    warpctc_grad,
    ops::WarpctcGradMLUKernel<paddle::platform::MLUDeviceContext, float>);
#endif
