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
#include "paddle/fluid/operators/warpctc_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CudnnCTCKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // =====================Copied code from warpctc===========================
    auto* logits = ctx.Input<LoDTensor>("Logits");
    auto* label = ctx.Input<LoDTensor>("Label");
    auto* warpctc_grad = ctx.Output<Tensor>("WarpCTCGrad");
    auto* loss = ctx.Output<Tensor>("Loss");

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

    const size_t sequence_width = logits->numel() / logits_dims[0];
    auto loss_dims =
        framework::make_ddim({static_cast<int64_t>(num_sequences), 1});

    // warpctc needs sequences data stored in transposed padding format
    Tensor warpctc_logits;
    const size_t max_sequence_length =
        math::MaximumSequenceLength(logits_lod, level);
    auto warpctc_logits_dims =
        framework::make_ddim({static_cast<int64_t>(max_sequence_length),
                              static_cast<int64_t>(num_sequences),
                              static_cast<int64_t>(sequence_width)});
    warpctc_logits.mutable_data<T>(warpctc_logits_dims, ctx.GetPlace());
    math::PaddingLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), *logits, &warpctc_logits,
        false);
    const T* warpctc_logits_data = warpctc_logits.data<T>();

    framework::Vector<int> warpctc_label_lengths(num_sequences);
    framework::Vector<int> warpctc_logits_lengths(num_sequences);

    for (size_t i = 0; i < num_sequences; ++i) {
      warpctc_label_lengths[i] = label_lod[level][i + 1] - label_lod[level][i];
      warpctc_logits_lengths[i] =
          logits_lod[level][i + 1] - logits_lod[level][i];
    }
    // ========================================================================
    // use name warpctc_grad_data for cudnn call for now, should be updated.
    T* warpctc_grad_data =
        warpctc_grad->mutable_data<T>(warpctc_logits.dims(), ctx.GetPlace());

    // TODO(typhoonzero): can remove setconstant?
    math::SetConstant<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), warpctc_grad,
        static_cast<T>(0));

    const int* label_data = label->data<int>();
    T* loss_data = loss->data<T>();

    const size_t blank = static_cast<size_t>(ctx.Attr<int>("blank"));

    // IMPORTANT: The labelLengths must less than 256 for cudnn call.
    ScopedTensorDescriptor logits_desc;
    ScopedTensorDescriptor grad_desc;
    ScopedCTCLossDescriptor ctcloss_desc;

    auto cu_logits_desc = logits_desc.descriptor<T>(
        layout, framework::vectorize2int(logits->dims()));
    auto cu_grad_desc = grad_desc.descriptor<T>(
        layout, framework::vectorize2int(warpctc_grad->dims()));
    auto cu_ctcloss_desc = ctcloss_desc.descriptor<T>(CUDNN_DATA_FLOAT);

    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    DataLayout layout = DataLayout::kNCHW;
    size_t workspace_size;

    CUDNN_ENFORCE(platform::dynload::cudnn::cudnnGetCTCLossWorkspaceSize(
        handle, cu_logits_desc, cu_grad_desc, label_data,
        warpctc_label_lengths.CUDAData(ctx.GetPlace()),
        warpctc_logits_lengths.CUDAData(ctx.GetPlace()),
        CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cu_ctcloss_desc, &workspace_size));

    void* workspace = memory::Alloc(ctx.GetPlace(), workspace_size);

    CUDNN_ENFORCE(platform::dynload::cudnn::cudnnCTCLoss(
        handle, cu_logits_desc, warpctc_logits_data, label_data,
        warpctc_label_lengths.CUDAData(ctx.GetPlace()),
        warpctc_logits_lengths.CUDAData(ctx.GetPlace()), loss_data,
        cu_grad_desc, warpctc_grad_data, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
        cu_ctcloss_desc, workspace, workspace_size));

    memory::Free(ctx.GetPlace(), workspace);
  }
};

template <typename DeviceContext, typename T>
class WarpCTCGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* warpctc_grad = ctx.Input<Tensor>("WarpCTCGrad");
    auto* logits_grad = ctx.Output<LoDTensor>(framework::GradVarName("Logits"));
    const Tensor* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));

    logits_grad->mutable_data<T>(ctx.GetPlace());
    bool norm_by_times = ctx.Attr<bool>("norm_by_times");
    math::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), logits_grad,
        *warpctc_grad, norm_by_times);

    const T* loss_grad_data = loss_grad->data<T>();
    math::ScaleLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), loss_grad_data,
        logits_grad);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    cudnnctc, ops::CudnnCTCKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    cudnnctc_grad,
    ops::CudnnCTCGradKernel<paddle::platform::CUDADeviceContext, float>);
