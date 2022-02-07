/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SoftmaxWithCrossEntropyMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* logits = ctx.Input<Tensor>("Logits");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* softmax = ctx.Output<Tensor>("Softmax");
    auto* loss = ctx.Output<Tensor>("Loss");
    auto* backprop = ctx.Output<Tensor>("Backprop");
    auto soft_label = ctx.Attr<bool>("soft_label");

    PADDLE_ENFORCE_EQ(ctx.Attr<bool>("use_softmax"), true,
                      platform::errors::InvalidArgument(
                          "use_softmax=False is not supported in "
                          "the mlu kernel of softmax_with_cross_entropy."));

    const int rank = logits->dims().size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);

    loss->mutable_data<T>(ctx.GetPlace());
    backprop->mutable_data<T>(ctx.GetPlace());
    softmax->mutable_data<T>(ctx.GetPlace());

    // cnnl softmax only support 3-dims, regard all shape as [d1, d2, d3]
    const int cnnl_softmax_dims = 3;
    const int d1 = SizeToAxis(axis, logits->dims());
    const int d2_logits = logits->dims()[axis];
    const int d2_labels = labels->dims()[axis];
    const int d3 = SizeOutAxis(axis, logits->dims());

    // CNNL_SOFTMAX_MODE_LOW_DIMENSION has better perfermence, use it as much as
    // possible.
    cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    std::vector<int> regard_logits_shape{d1, 1, d2_logits};
    std::vector<int> regard_labels_shape{d1, 1, d2_labels};
    std::vector<int> regard_loss_shape{d1, 1, 1};
    if (d3 != 1) {
      mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
      regard_logits_shape = {d1, d2_logits, d3};
      regard_labels_shape = {d1, d2_labels, d3};
      regard_loss_shape = {d1, 1, d3};
    }

    MLUCnnlTensorDesc logits_desc(cnnl_softmax_dims, regard_logits_shape.data(),
                                  ToCnnlDataType<T>());
    MLUCnnlTensorDesc labels_desc(cnnl_softmax_dims, regard_labels_shape.data(),
                                  ToCnnlDataType<T>());
    MLUCnnlTensorDesc loss_desc(cnnl_softmax_dims, regard_loss_shape.data(),
                                ToCnnlDataType<T>());

    const cnnlSoftmaxAlgorithm_t algo = CNNL_SOFTMAX_ACCURATE;
    MLUCnnl::SoftmaxForward(ctx, algo, mode, NULL, logits_desc.get(),
                            GetBasePtr(logits), NULL, logits_desc.get(),
                            GetBasePtr(softmax));

    if (soft_label) {
      const cnnlComputationPreference_t prefer =
          CNNL_COMPUTATION_HIGH_PRECISION;
      MLUCnnl::SoftmaxCrossEntropyWithLogits(
          ctx, mode, prefer, logits_desc.get(), GetBasePtr(logits),
          labels_desc.get(), GetBasePtr(labels), loss_desc.get(),
          GetBasePtr(loss), logits_desc.get(), GetBasePtr(backprop));
    } else {
      PADDLE_ENFORCE_EQ(d3, 1,
                        platform::errors::InvalidArgument(
                            "If soft_label=False, axis must be -1 or"
                            " can be regard as last dimention in mlu kernel."));
      framework::Tensor labels_int32(VT::INT32);
      labels_int32.Resize(labels->dims());
      labels_int32.mutable_data<int32_t>(ctx.GetPlace());

      MLUCnnlTensorDesc labels_int64_desc(*labels);
      MLUCnnlTensorDesc labels_int32_desc(labels_int32);
      cnnlCastDataType_t cast_type = GetCastDataType(VT::INT64, VT::INT32);
      MLUCnnl::Cast(ctx, cast_type, labels_int64_desc.get(), GetBasePtr(labels),
                    labels_int32_desc.get(), GetBasePtr(&labels_int32));

      const int regard_sparse_shape[cnnl_softmax_dims - 1] = {d1, 1};
      MLUCnnlTensorDesc sparse_labels_desc(cnnl_softmax_dims - 1,
                                           regard_sparse_shape,
                                           ToCnnlDataType<int32_t>());
      MLUCnnlTensorDesc sparse_loss_desc(
          cnnl_softmax_dims - 1, regard_sparse_shape, ToCnnlDataType<T>());

      MLUCnnl::SparseSoftmaxXentWithLogits(
          ctx, mode, logits_desc.get(), GetBasePtr(logits),
          sparse_labels_desc.get(), GetBasePtr(&labels_int32),
          sparse_loss_desc.get(), GetBasePtr(loss), logits_desc.get(),
          GetBasePtr(backprop));
    }
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* backprop = ctx.Input<Tensor>("Backprop");
    auto* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto* logits_grad = ctx.Output<Tensor>(framework::GradVarName("Logits"));
    PADDLE_ENFORCE_NOT_NULL(backprop,
                            platform::errors::PreconditionNotMet(
                                "backprop should not be null in MLU kernel of "
                                "softmax_with_cross_entropy_grad."));
    logits_grad->mutable_data<T>(ctx.GetPlace());

    MLUCnnlOpTensorDesc mul_op_desc(CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(),
                                    CNNL_NOT_PROPAGATE_NAN);
    MLUCnnlTensorDesc backprop_desc(*backprop);
    MLUCnnlTensorDesc loss_grad_desc(*loss_grad);
    MLUCnnlTensorDesc logits_grad_desc(*logits_grad);
    MLUCnnl::OpTensor(ctx, mul_op_desc.get(), backprop_desc.get(),
                      GetBasePtr(backprop), loss_grad_desc.get(),
                      GetBasePtr(loss_grad), logits_grad_desc.get(),
                      GetBasePtr(logits_grad), ToCnnlDataType<T>());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyMLUKernel<float>,
    ops::SoftmaxWithCrossEntropyMLUKernel<paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradMLUKernel<float>,
    ops::SoftmaxWithCrossEntropyGradMLUKernel<paddle::platform::float16>);
