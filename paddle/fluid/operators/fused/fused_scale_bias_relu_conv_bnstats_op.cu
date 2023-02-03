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

#include <array>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/operators/layout_utils.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_desc.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using DataLayout = platform::DataLayout;
using helper = phi::CudnnFrontendConvHelper;
using TensorDesc_t = cudnn_frontend::Tensor;

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T>
class FusedScaleBiasReluConvBnstatsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    // attr
    bool fuse_prologue = ctx.Attr<bool>("fuse_prologue");
    // input variables
    auto* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    // output variables
    using U = BatchNormParamType<T>;
    auto* output = ctx.Output<Tensor>("Output");
    auto* sum_output = ctx.Output<Tensor>("SumOutput");
    auto* sqsum_output = ctx.Output<Tensor>("SqSumOutput");
    output->mutable_data<T>(ctx.GetPlace());
    sum_output->mutable_data<U>(ctx.GetPlace());
    sqsum_output->mutable_data<U>(ctx.GetPlace());
    // deal with strides, dilations and paddings
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");

    // transformed tensor
    Tensor transformed_input_channel(input->dtype());
    Tensor transformed_output(output->dtype());
    Tensor transformed_filter_channel(filter->dtype());
    // Assume input and output already in NHWC.
    // No transformation is needed for them.
    transformed_input_channel = *input;
    transformed_output = *output;

    using Context = phi::GPUContext;
    VLOG(3) << "Transform filter tensor from NCHW to NHWC.";
    ResizeToChannelLast<Context, T>(ctx, filter, &transformed_filter_channel);
    TransToChannelLast<Context, T>(ctx, filter, &transformed_filter_channel);

    // update padding and dilation
    auto in_dims = transformed_input_channel.dims();
    auto filter_dims = transformed_filter_channel.dims();
    framework::DDim in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
    framework::DDim filter_data_dims =
        slice_ddim(filter_dims, 1, filter_dims.size() - 1);
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    int data_dim = strides.size();  // 2d or 3d

    Tensor transformed_input;
    std::vector<int64_t> pre_padding(data_dim, 0);
    std::vector<int64_t> post_padding(data_dim, 0);
    transformed_input = transformed_input_channel;
    for (size_t i = 0; i < data_dim; ++i) {
      pre_padding[i] = static_cast<int64_t>(paddings[2 * i]);
      post_padding[i] = static_cast<int64_t>(paddings[2 * i + 1]);
    }

    // input pointers
    T* input_data = transformed_input.data<T>();
    T* filter_data = transformed_filter_channel.data<T>();

    // output pointers
    T* output_data = transformed_output.data<T>();
    U* sum_output_data = sum_output->data<U>();
    U* sqsum_output_data = sqsum_output->data<U>();

    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();

    const int rank = transformed_input_channel.dims().size();

    // build tensors
    cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
    auto tensor_format =
        phi::backends::gpu::ToCudnnDataType(transformed_input.dtype());
    auto tensor_format_math = CUDNN_DATA_FLOAT;
    auto compute_dtype = CUDNN_DATA_FLOAT;

    // get dims in CUDNN manner: [N, C, H, W]
    auto dim_x =
        phi::backends::gpu::TransformDimOrder(phi::vectorize<int64_t>(in_dims));
    auto dim_filt = phi::backends::gpu::TransformDimOrder(
        phi::vectorize<int64_t>(filter_dims));
    auto dim_y = phi::backends::gpu::TransformDimOrder(
        phi::vectorize<int64_t>(output->dims()));
    std::vector<int64_t> dim_scale(dim_x.size(), 1);
    dim_scale[1] = dim_x[1];                        //  [1, C, 1, 1]
    std::vector<int64_t> dim_sum(dim_x.size(), 1);  // [1, K, 1, 1]
    dim_sum[1] = dim_filt[0];

    std::vector<void*> data_ptrs;
    std::vector<int64_t> uids;
    int64_t uid = 100;

    // inputs
    auto input_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(input_data);
    uids.push_back(uid++);

    auto filter_desc = helper::GetGeneralTensorDescriptor(
        dim_filt, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(filter_data);
    uids.push_back(uid++);

    // dispensable inputs
    auto scale_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format);
    if (fuse_prologue) {
      auto* scale = ctx.Input<Tensor>("Scale");
      data_ptrs.push_back(const_cast<T*>(scale->data<T>()));
      uids.push_back(uid);
    }
    uid++;

    auto bias_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format);
    if (fuse_prologue) {
      auto* bias = ctx.Input<Tensor>("Bias");
      data_ptrs.push_back(const_cast<T*>(bias->data<T>()));
      uids.push_back(uid);
    }
    uid++;

    // outputs
    auto output_desc = helper::GetGeneralTensorDescriptor(
        dim_y, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(output_data);
    uids.push_back(uid++);

    auto sum_output_desc = helper::GetGeneralTensorDescriptor(
        dim_sum, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(sum_output_data);
    uids.push_back(uid++);

    auto sqsum_output_desc = helper::GetGeneralTensorDescriptor(
        dim_sum, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(sqsum_output_data);
    uids.push_back(uid++);

    // virtual outputs
    auto after_scale = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);
    auto after_bias = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);
    auto after_relu = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    // create ops
    auto scale_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                            compute_dtype,
                                            input_desc,
                                            scale_desc,
                                            after_scale);

    auto bias_op = helper::MakePointwiseOp(
        CUDNN_POINTWISE_ADD, compute_dtype, after_scale, bias_desc, after_bias);

    auto relu_desc = cudnn_frontend::PointWiseDescBuilder()
                         .setMode(CUDNN_POINTWISE_RELU_FWD)
                         .setComputeType(compute_dtype)
                         .build();

    auto relu_op = cudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                       .setxDesc(after_bias)
                       .setyDesc(after_relu)
                       .setpwDesc(relu_desc)
                       .build();
    VLOG(6) << relu_op.describe();

    std::vector<int64_t> stride_int64 = helper::GetInt64Array(strides);
    std::vector<int64_t> dilation_int64 = helper::GetInt64Array(dilations);
    auto conv_desc = cudnn_frontend::ConvDescBuilder()
                         .setComputeType(compute_dtype)
                         .setMathMode(CUDNN_CROSS_CORRELATION)
                         .setSpatialDimCount(data_dim)
                         .setSpatialStride(data_dim, stride_int64.data())
                         .setPrePadding(data_dim, pre_padding.data())
                         .setPostPadding(data_dim, post_padding.data())
                         .setDilation(data_dim, dilation_int64.data())
                         .build();

    float alpha = 1.0f;
    float beta = 0.0f;
    TensorDesc_t* input_to_conv = fuse_prologue ? &after_relu : &input_desc;
    auto conv_op = cudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                       .setxDesc(*input_to_conv)
                       .setwDesc(filter_desc)
                       .setyDesc(output_desc)
                       .setcDesc(conv_desc)
                       .setAlpha(alpha)
                       .setBeta(beta)
                       .build();
    VLOG(6) << conv_op.describe();

    auto genstat_op = cudnn_frontend::OperationBuilder(
                          CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR)
                          .setxDesc(output_desc)
                          .setComputeType(compute_dtype)
                          .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
                          .setSumDesc(sum_output_desc)
                          .setSqSumDesc(sqsum_output_desc)
                          .build();
    VLOG(6) << genstat_op.describe();

    // build op graph
    std::vector<cudnn_frontend::Operation const*> ops;
    if (fuse_prologue) {
      ops = std::vector<cudnn_frontend::Operation const*>(
          {&scale_op, &bias_op, &relu_op, &conv_op, &genstat_op});
    } else {
      ops = std::vector<cudnn_frontend::Operation const*>(
          {&conv_op, &genstat_op});
    }

    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                        .setHandle(handle)
                        .setOperationGraph(ops.size(), ops.data())
                        .build();
    VLOG(6) << op_graph.describe();

    auto plan = helper::GetPlanByHeuristics(std::move(op_graph), handle);
    auto workspace_size = plan.getWorkspaceSize();
    VLOG(4) << plan.describe() << " requires workspace " << workspace_size;

    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        plan.get_raw_desc(),
                        workspace_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_scale_bias_relu_conv_bnstats,
    ops::FusedScaleBiasReluConvBnstatsOpKernel<paddle::platform::float16>);
