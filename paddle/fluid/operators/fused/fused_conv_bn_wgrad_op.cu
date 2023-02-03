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
#include <memory>

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
using ManagedOpaqueDescriptor = cudnn_frontend::ManagedOpaqueDescriptor;
using OpaqueBackendPointer = cudnn_frontend::OpaqueBackendPointer;

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
class FusedConvBnWgradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    // required input variables
    auto* bn_x_tensor = ctx.Input<Tensor>("BN_X");
    auto* dy_tensor = ctx.Input<Tensor>("dY");
    auto* scale_tensor = ctx.Input<Tensor>("Scale");
    auto* bias_tensor = ctx.Input<Tensor>("Bias");
    // required output variables
    auto* dw_tensor = ctx.Output<Tensor>("dW");
    dw_tensor->mutable_data<T>(ctx.GetPlace());
    // transform filter to NHWC layout
    Tensor dw_tensor_transformed(dw_tensor->dtype());
    using Context = phi::GPUContext;
    ResizeToChannelLast<Context, T>(ctx, dw_tensor, &dw_tensor_transformed);
    // deal with strides, dilations and paddings
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    // update padding and dilation
    auto in_dims = bn_x_tensor->dims();
    auto filter_dims = dw_tensor_transformed.dims();
    framework::DDim in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
    framework::DDim filter_data_dims =
        slice_ddim(filter_dims, 1, filter_dims.size() - 1);
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);
    int data_dim = strides.size();  // 2d or 3d
    std::vector<int64_t> pre_padding(data_dim, 0);
    std::vector<int64_t> post_padding(data_dim, 0);
    for (size_t i = 0; i < data_dim; ++i) {
      pre_padding[i] = static_cast<int64_t>(paddings[2 * i]);
      post_padding[i] = static_cast<int64_t>(paddings[2 * i + 1]);
    }
    // get handles
    auto handle = dev_ctx.cudnn_handle();
    auto workspace_handle = dev_ctx.cudnn_workspace_handle();
    // create tensor descriptors
    cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
    auto tensor_format =
        phi::backends::gpu::ToCudnnDataType(bn_x_tensor->dtype());
    auto tensor_format_math = CUDNN_DATA_FLOAT;
    auto compute_dtype = CUDNN_DATA_FLOAT;

    auto dim_x =
        phi::backends::gpu::TransformDimOrder(phi::vectorize<int64_t>(in_dims));
    auto dim_filt = phi::backends::gpu::TransformDimOrder(
        phi::vectorize<int64_t>(filter_dims));
    auto dim_y = phi::backends::gpu::TransformDimOrder(
        phi::vectorize<int64_t>(dy_tensor->dims()));
    std::vector<int64_t> dim_scale(dim_x.size(), 1);
    dim_scale[1] = dim_x[1];  //  [1, C, 1, 1]

    std::vector<void*> data_ptrs;
    std::vector<int64_t> uids;
    int64_t uid = 100;

    // inputs
    auto bn_x_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(bn_x_tensor->data<T>()));
    uids.push_back(uid++);

    auto dy_desc = helper::GetGeneralTensorDescriptor(
        dim_y, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(dy_tensor->data<T>()));
    uids.push_back(uid++);

    auto scale_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(scale_tensor->data<T>()));
    uids.push_back(uid++);

    auto bias_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(bias_tensor->data<T>()));
    uids.push_back(uid++);

    // outputs
    auto dw_desc = helper::GetGeneralTensorDescriptor(
        dim_filt, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(dw_tensor_transformed.data<T>());
    uids.push_back(uid++);

    // virtual outputs
    auto after_scale = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    auto after_bias = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    auto after_relu = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    // build ops
    std::vector<int64_t> stride_int64 = helper::GetInt64Array(strides);
    std::vector<int64_t> dilation_int64 = helper::GetInt64Array(dilations);
    auto conv_desc = cudnn_frontend::ConvDescBuilder()
                         .setComputeType(CUDNN_DATA_FLOAT)
                         .setMathMode(CUDNN_CROSS_CORRELATION)
                         .setSpatialDimCount(data_dim)
                         .setSpatialStride(data_dim, stride_int64.data())
                         .setPrePadding(data_dim, pre_padding.data())
                         .setPostPadding(data_dim, post_padding.data())
                         .setDilation(data_dim, dilation_int64.data())
                         .build();
    VLOG(6) << conv_desc.describe();

    auto wgrad_op =
        cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
            .setdyDesc(dy_desc)
            .setdwDesc(dw_desc)
            .setxDesc(after_relu)
            .setcDesc(conv_desc)
            .setAlpha(1.0f)
            .setBeta(0.0f)
            .build();
    VLOG(6) << wgrad_op.describe();

    auto scale_op = helper::MakePointwiseOp(
        CUDNN_POINTWISE_MUL, compute_dtype, bn_x_desc, scale_desc, after_scale);

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

    // build op graph
    std::array<cudnn_frontend::Operation const*, 4> ops = {
        &scale_op, &bias_op, &relu_op, &wgrad_op};

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

    // transfer back to NCWH
    TransToChannelFirst<Context, T>(ctx, &dw_tensor_transformed, dw_tensor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_conv_bn_wgrad,
    ops::FusedConvBnWgradOpKernel<paddle::platform::float16>);
