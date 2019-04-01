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

#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/warpctc_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

#if CUDNN_VERSION >= 7001
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using ScopedCTCLossDescriptor = platform::ScopedCTCLossDescriptor;
using DataLayout = platform::DataLayout;

template <typename DeviceContext, typename T>
class CudnnCTCKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // =====================Copied code from warpctc===========================
    auto* logits = ctx.Input<LoDTensor>("Logits");
    auto* label = ctx.Input<LoDTensor>("Label");
    auto* warpctc_grad = ctx.Output<LoDTensor>("WarpCTCGrad");
    auto* loss = ctx.Output<LoDTensor>("Loss");

    const size_t level = 0;

    auto logits_lod = framework::ToAbsOffset(logits->lod());
    auto logits_dims = logits->dims();
    PADDLE_ENFORCE_EQ(logits_dims[0],
                      static_cast<int64_t>(logits_lod[level].back()),
                      "The first dimension of Input(Logits) should be equal to "
                      "the sum of all sequences' lengths.");

    auto label_lod = framework::ToAbsOffset(label->lod());
    auto label_dims = label->dims();
    PADDLE_ENFORCE_EQ(
        label_dims[0], label->numel(),
        "The width of each timestep in Input(Label) should be 1.");

    const size_t num_sequences = logits_lod[level].size() - 1;
    PADDLE_ENFORCE_EQ(num_sequences, label_lod[level].size() - 1,
                      "The number of sequences of Input(Logits) should be "
                      "equal to that of Input(Label).");
    PADDLE_ENFORCE_LE(num_sequences, 256,
                      "The labelLengths must less than 256 for cudnn call.");

    const size_t sequence_width = logits->numel() / logits_dims[0];
    auto loss_dims =
        framework::make_ddim({static_cast<int64_t>(num_sequences), 1});

    // NOTE: cudnn takes softmax input, calculate softmax first, then do padding
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    LoDTensor softmax_logits;
    softmax_logits.mutable_data<T>(logits->dims(), ctx.GetPlace());
    softmax_logits.set_lod(logits_lod);
    int rank = logits->dims().size();
    int axis_dim = logits->dims()[rank - 1];
    Tensor in_2d = framework::ReshapeToMatrix(*logits, rank - 1);
    Tensor out_2d = framework::ReshapeToMatrix(softmax_logits, rank - 1);
    math::SoftmaxFunctor<DeviceContext, T, false>()(dev_ctx, axis_dim, &in_2d,
                                                    &out_2d);

    // ctc needs sequences data stored in transposed padding format
    // logits and grad using padding data of layout 'TNC'
    // T: max_sequence_length
    // N: batch_size (num_sequences)
    // C: width
    LoDTensor warpctc_logits;
    const size_t max_sequence_length =
        math::MaximumSequenceLength(logits_lod[level]);
    auto warpctc_logits_dims =
        framework::make_ddim({static_cast<int64_t>(max_sequence_length),
                              static_cast<int64_t>(num_sequences),
                              static_cast<int64_t>(sequence_width)});
    warpctc_logits.mutable_data<T>(warpctc_logits_dims, ctx.GetPlace());

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
        ctx.template device_context<DeviceContext>(), softmax_logits,
        &warpctc_logits, pad_value, -1, 0, false /* norm_by_times */,
        math::kLengthBatchWidth);
    const T* warpctc_logits_data = warpctc_logits.data<T>();

    std::vector<int> warpctc_label_lengths(num_sequences);
    std::vector<int> warpctc_logits_lengths(num_sequences);

    for (size_t i = 0; i < num_sequences; ++i) {
      warpctc_label_lengths[i] = label_lod[level][i + 1] - label_lod[level][i];
      warpctc_logits_lengths[i] =
          logits_lod[level][i + 1] - logits_lod[level][i];
    }

    T* warpctc_grad_data =
        warpctc_grad->mutable_data<T>(warpctc_logits.dims(), ctx.GetPlace());

    math::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), warpctc_grad,
        static_cast<T>(0));

    Tensor warpctc_label;
    TensorCopySync(*label, platform::CPUPlace(), &warpctc_label);
    const int* warpctc_label_data = warpctc_label.data<int>();
    // ========================================================================

    ScopedTensorDescriptor logits_desc;
    ScopedTensorDescriptor grad_desc;
    ScopedCTCLossDescriptor ctcloss_desc;
    // layout here doesn't have effect.
    DataLayout layout = DataLayout::kNCHW;

    auto cu_logits_desc = logits_desc.descriptor<T>(
        layout, framework::vectorize2int(warpctc_logits.dims()));
    auto cu_grad_desc = grad_desc.descriptor<T>(
        layout, framework::vectorize2int(warpctc_grad->dims()));
    auto cu_ctcloss_desc = ctcloss_desc.descriptor<T>();

    auto handle = dev_ctx.cudnn_handle();
    size_t workspace_size;

    CUDNN_ENFORCE(platform::dynload::cudnnGetCTCLossWorkspaceSize(
        handle, cu_logits_desc, cu_grad_desc, warpctc_label_data,
        warpctc_label_lengths.data(), warpctc_logits_lengths.data(),
        CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cu_ctcloss_desc, &workspace_size));

    T* loss_data = loss->mutable_data<T>(loss_dims, ctx.GetPlace());

    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    auto cudnn_func = [&](void* cudnn_workspace) {
      CUDNN_ENFORCE(platform::dynload::cudnnCTCLoss(
          handle, cu_logits_desc, warpctc_logits_data, warpctc_label_data,
          warpctc_label_lengths.data(), warpctc_logits_lengths.data(),
          loss_data, cu_grad_desc, warpctc_grad_data,
          CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cu_ctcloss_desc, cudnn_workspace,
          workspace_size));
    };
    workspace_handle.RunFunc(cudnn_func, workspace_size);
  }
};

template <typename DeviceContext, typename T>
class CudnnCTCGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* warpctc_grad = ctx.Input<LoDTensor>("WarpCTCGrad");
    auto* logits_grad = ctx.Output<LoDTensor>(framework::GradVarName("Logits"));
    const Tensor* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));

    logits_grad->mutable_data<T>(ctx.GetPlace());
    bool norm_by_times = ctx.Attr<bool>("norm_by_times");
    math::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), *warpctc_grad,
        logits_grad, -1, 0, norm_by_times, math::kLengthBatchWidth);

    const T* loss_grad_data = loss_grad->data<T>();
    math::ScaleLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), loss_grad_data,
        logits_grad);
  }
};

#endif
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#if CUDNN_VERSION >= 7001
REGISTER_OP_KERNEL(
    warpctc, CUDNN, plat::CUDAPlace,
    ops::CudnnCTCKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_KERNEL(
    warpctc_grad, CUDNN, plat::CUDAPlace,
    ops::CudnnCTCGradKernel<paddle::platform::CUDADeviceContext, float>);
#endif
