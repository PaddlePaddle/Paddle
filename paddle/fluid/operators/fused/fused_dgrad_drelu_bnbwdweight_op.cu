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
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/backends/dynload/cudnn.h"
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
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

namespace {
cudnn_frontend::Operation MakeDreluOp(cudnnDataType_t dtype,
                                      TensorDesc_t const& dy_desc,
                                      TensorDesc_t const& x_desc,
                                      TensorDesc_t const& dx_desc) {
  auto op_desc = cudnn_frontend::PointWiseDescBuilder()
                     .setMode(CUDNN_POINTWISE_RELU_BWD)
                     .setComputeType(dtype)
                     .build();
  auto op = cudnn_frontend::OperationBuilder(
                CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setdyDesc(dy_desc)
                .setxDesc(x_desc)
                .setdxDesc(dx_desc)
                .setpwDesc(op_desc)
                .build();
  VLOG(6) << op.describe();
  return op;
}
// BnBwdWeight is not supported by CUDNN Frontend API yet
// Use backend API instead
// This is considered a temporary workaround
ManagedOpaqueDescriptor BackendMakeBnbwdweightOp(
    cudnnDataType_t dtype,
    TensorDesc_t const& x_desc,
    TensorDesc_t const& mean_desc,
    TensorDesc_t const& invstd_desc,
    TensorDesc_t const& bn_scale_desc,
    TensorDesc_t const& dy_desc,
    TensorDesc_t const& dbn_bias_desc,
    TensorDesc_t const& dbn_scale_desc,
    TensorDesc_t const& eq_dy_scale_desc,
    TensorDesc_t const& eq_x_scale_desc,
    TensorDesc_t const& eqbias_desc) {
  ManagedOpaqueDescriptor pointer;
  pointer = std::make_shared<OpaqueBackendPointer>(
      CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR);
  PADDLE_ENFORCE_GPU_SUCCESS(pointer->get_status());
  cudnnBackendDescriptor_t op_desc = pointer->get_backend_descriptor();

  auto x_desc_raw = x_desc.get_raw_desc();
  auto mean_desc_raw = mean_desc.get_raw_desc();
  auto invstd_desc_raw = invstd_desc.get_raw_desc();
  auto bn_scale_desc_raw = bn_scale_desc.get_raw_desc();
  auto dy_desc_raw = dy_desc.get_raw_desc();
  auto dbn_bias_desc_raw = dbn_bias_desc.get_raw_desc();
  auto dbn_scale_desc_raw = dbn_scale_desc.get_raw_desc();
  auto eq_dy_scale_desc_raw = eq_dy_scale_desc.get_raw_desc();
  auto eq_x_scale_desc_raw = eq_x_scale_desc.get_raw_desc();
  auto eqbias_desc_raw = eqbias_desc.get_raw_desc();

  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC,
      CUDNN_TYPE_DATA_TYPE,
      1,
      &(dtype)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(x_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(mean_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(invstd_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(bn_scale_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(dy_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(dbn_bias_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(dbn_scale_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(eq_dy_scale_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(eq_x_scale_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
      op_desc,
      CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS,
      CUDNN_TYPE_BACKEND_DESCRIPTOR,
      1,
      &(eqbias_desc_raw)));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendFinalize(op_desc));
  return pointer;
}

ManagedOpaqueDescriptor BackendBuildOperationGraph(
    cudnnHandle_t handle, cudnnBackendDescriptor_t* ops, int len) {
  ManagedOpaqueDescriptor op_graph = nullptr;
  op_graph = std::make_shared<OpaqueBackendPointer>(
      CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  PADDLE_ENFORCE_GPU_SUCCESS(op_graph->get_status());
  cudnnBackendDescriptor_t op_graph_desc = op_graph->get_backend_descriptor();
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnBackendSetAttribute(op_graph_desc,
                                             CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                             CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                             len,
                                             ops));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cudnnBackendSetAttribute(op_graph_desc,
                                             CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                             CUDNN_TYPE_HANDLE,
                                             1,
                                             &handle));
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendFinalize(op_graph_desc));
  return op_graph;
}

// This is considered a temporary workaround.
// Will be removed after BnBwdWeight is supported by Frontend API
class BackendExecutionPlan {
 public:
  BackendExecutionPlan() = default;
  ~BackendExecutionPlan() = default;
  cudnnBackendDescriptor_t GetDesc() {
    PADDLE_ENFORCE_NE(
        plan,
        nullptr,
        platform::errors::InvalidArgument("Plan pointer is nullptr."));
    return plan->get_backend_descriptor();
  }

  int64_t GetWorkspaceSize() {
    int64_t workspace_size;
    PADDLE_ENFORCE_NE(
        plan,
        nullptr,
        platform::errors::InvalidArgument("Plan pointer is nullptr."));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendGetAttribute(
        plan->get_backend_descriptor(),
        CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
        CUDNN_TYPE_INT64,
        1,
        nullptr,
        &workspace_size));
    return workspace_size;
  }

  cudnnStatus_t MakePlan(cudnnHandle_t handle,
                         int64_t engine_id,
                         cudnnBackendDescriptor_t op_graph) {
    // make engine descriptor
    engine =
        std::make_shared<OpaqueBackendPointer>(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    PADDLE_ENFORCE_GPU_SUCCESS(engine->get_status());
    cudnnBackendDescriptor_t engine_desc = engine->get_backend_descriptor();
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
        engine_desc,
        CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
        CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        &op_graph));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnBackendSetAttribute(engine_desc,
                                               CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                               CUDNN_TYPE_INT64,
                                               1,
                                               &engine_id));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendFinalize(engine_desc));

    // make engine config descriptor
    engine_cfg = std::make_shared<OpaqueBackendPointer>(
        CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    PADDLE_ENFORCE_GPU_SUCCESS(engine_cfg->get_status());
    cudnnBackendDescriptor_t engine_cfg_desc =
        engine_cfg->get_backend_descriptor();

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnBackendSetAttribute(engine_cfg_desc,
                                               CUDNN_ATTR_ENGINECFG_ENGINE,
                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                               1,
                                               &engine_desc));
    // PADDLE_ENFORCE_GPU_SUCCESS(cudnnBackendSetAttribute(
    //     engine_cfg, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
    //     CUDNN_TYPE_BACKEND_DESCRIPTOR, numKnobs, &choices[0]));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnBackendFinalize(engine_cfg_desc));

    // make plan
    plan = std::make_shared<OpaqueBackendPointer>(
        CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    PADDLE_ENFORCE_GPU_SUCCESS(plan->get_status());
    cudnnBackendDescriptor_t plan_desc = plan->get_backend_descriptor();

    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendSetAttribute(
        plan_desc,
        CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        CUDNN_TYPE_BACKEND_DESCRIPTOR,
        1,
        &engine_cfg_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnBackendSetAttribute(plan_desc,
                                               CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                               CUDNN_TYPE_HANDLE,
                                               1,
                                               &handle));
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnBackendFinalize(plan_desc));

    return CUDNN_STATUS_SUCCESS;
  }

 private:
  ManagedOpaqueDescriptor engine = nullptr;
  ManagedOpaqueDescriptor engine_cfg = nullptr;
  ManagedOpaqueDescriptor plan = nullptr;
};  // class BackendExecutionPlan
}  // namespace

template <typename T>
class FusedDgradDreluBnBwdWeightOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = BatchNormParamType<T>;
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    auto cudnn_version = platform::DnnVersion();
    PADDLE_ENFORCE_EQ(
        cudnn_version < 8500 || cudnn_version >= 8800,
        true,
        phi::errors::PreconditionNotMet(
            "This op only supports CUDNN version >= 8800 or < 8500, "
            "but got %d.",
            cudnn_version));
    // Attributes
    bool fuse_shortcut = ctx.Attr<bool>("fuse_shortcut");
    bool fuse_dual = ctx.Attr<bool>("fuse_dual");
    bool fuse_add = ctx.Attr<bool>("fuse_add");
    // required input variables
    const Tensor* dy_tensor = ctx.Input<Tensor>("dY");
    const Tensor* w_tensor = ctx.Input<Tensor>("W");
    const Tensor* bn1_mean_tensor = ctx.Input<Tensor>("BN1_mean");
    const Tensor* bn1_inv_std_tensor = ctx.Input<Tensor>("BN1_inv_std");
    const Tensor* bn1_scale_tensor = ctx.Input<Tensor>("BN1_scale");
    const Tensor* bn1_bias_tensor = ctx.Input<Tensor>("BN1_bias");
    const Tensor* bn1_x_tensor = ctx.Input<Tensor>("BN1_X");
    // required output variables
    Tensor* dx_tensor = ctx.Output<Tensor>("dX");
    Tensor* bn1_dgamma_tensor = ctx.Output<Tensor>("BN1_dGamma");
    Tensor* bn1_dbeta_tensor = ctx.Output<Tensor>("BN1_dBeta");
    Tensor* bn1_dBeta_tensor = ctx.Output<Tensor>("BN1_eqscale_dy");
    Tensor* bn1_eqscale_x_tensor = ctx.Output<Tensor>("BN1_eqscale_x");
    Tensor* bn1_eqbias_tensor = ctx.Output<Tensor>("BN1_eqbias");
    dx_tensor->mutable_data<T>(ctx.GetPlace());
    bn1_dgamma_tensor->mutable_data<U>(ctx.GetPlace());
    bn1_dbeta_tensor->mutable_data<U>(ctx.GetPlace());
    bn1_dBeta_tensor->mutable_data<U>(ctx.GetPlace());
    bn1_eqscale_x_tensor->mutable_data<U>(ctx.GetPlace());
    bn1_eqbias_tensor->mutable_data<U>(ctx.GetPlace());
    // transform filter to NHWC layout
    Tensor w_tensor_transformed(w_tensor->dtype());
    using Context = phi::GPUContext;
    ResizeToChannelLast<Context, T>(ctx, w_tensor, &w_tensor_transformed);
    TransToChannelLast<Context, T>(ctx, w_tensor, &w_tensor_transformed);
    // deal with strides, dilations and paddings
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    // update padding and dilation
    auto in_dims = bn1_x_tensor->dims();
    auto filter_dims = w_tensor_transformed.dims();
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
    // build tensor descriptors
    cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
    auto tensor_format =
        phi::backends::gpu::ToCudnnDataType(dy_tensor->dtype());
    auto tensor_format_math = CUDNN_DATA_FLOAT;
    auto compute_dtype = CUDNN_DATA_FLOAT;

    // get dims in CUDNN manner: [N, C, H, W]
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

    // dgrad inputs
    auto dy_desc = helper::GetGeneralTensorDescriptor(
        dim_y, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(dy_tensor->data<T>()));
    uids.push_back(uid++);

    auto w_desc = helper::GetGeneralTensorDescriptor(
        dim_filt, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(w_tensor_transformed.data<T>()));
    uids.push_back(uid++);

    // dBN1 inputs
    auto bn1_mean_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(const_cast<U*>(bn1_mean_tensor->data<U>()));
    uids.push_back(uid++);

    auto bn1_inv_std_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(const_cast<U*>(bn1_inv_std_tensor->data<U>()));
    uids.push_back(uid++);

    auto bn1_scale_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(const_cast<U*>(bn1_scale_tensor->data<U>()));
    uids.push_back(uid++);

    auto bn1_bias_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(const_cast<U*>(bn1_bias_tensor->data<U>()));
    uids.push_back(uid++);

    auto bn1_x_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(const_cast<T*>(bn1_x_tensor->data<T>()));
    uids.push_back(uid++);

    // fuse_add inputs
    auto dx_branch_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid, 16, tensor_format);
    if (fuse_add) {
      auto* dx_branch_tensor = ctx.Input<Tensor>("dX_branch");
      data_ptrs.push_back(const_cast<T*>(dx_branch_tensor->data<T>()));
      uids.push_back(uid);
    }
    uid++;

    // virtual outputs
    auto dx_dgrad_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);
    auto after_add0 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);
    auto after_add1 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);
    auto after_mul1 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);
    auto after_add2 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);
    auto after_mul2 = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid++, 16, tensor_format_math, true);

    // drelu outputs
    auto dx_desc = helper::GetGeneralTensorDescriptor(
        dim_x, layout_format, uid, 16, tensor_format);
    data_ptrs.push_back(dx_tensor->data<T>());
    uids.push_back(uid++);

    // dBN1 outputs
    auto bn1_dgamma_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(bn1_dgamma_tensor->data<U>());
    uids.push_back(uid++);

    auto bn1_dbeta_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(bn1_dbeta_tensor->data<U>());
    uids.push_back(uid++);

    auto bn1_eqscale_dy_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(bn1_dBeta_tensor->data<U>());
    uids.push_back(uid++);

    auto bn1_eqscale_x_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(bn1_eqscale_x_tensor->data<U>());
    uids.push_back(uid++);

    auto bn1_eqbias_desc = helper::GetGeneralTensorDescriptor(
        dim_scale, layout_format, uid, 16, tensor_format_math);
    data_ptrs.push_back(bn1_eqbias_tensor->data<U>());
    uids.push_back(uid++);

    // build desc
    std::vector<cudnnBackendDescriptor_t> backend_ops;
    // make dgrad op
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

    auto dgrad_op =
        cudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
            .setdyDesc(dy_desc)
            .setwDesc(w_desc)
            .setdxDesc(dx_dgrad_desc)
            .setcDesc(conv_desc)
            .setAlpha(1.0f)
            .setBeta(0.0f)
            .build();
    VLOG(6) << dgrad_op.describe();
    backend_ops.push_back(dgrad_op.get_raw_desc());

    TensorDesc_t* p_drelu_input_desc = &dx_dgrad_desc;
    auto add0_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                           compute_dtype,
                                           dx_dgrad_desc,
                                           dx_branch_desc,
                                           after_add0);
    if (fuse_add) {
      backend_ops.push_back(add0_op.get_raw_desc());
      p_drelu_input_desc = &after_add0;
    }
    // make pointwise nodes
    if (fuse_shortcut) {
      // additional drelu inputs (optional)
      auto* relu_x_tensor = ctx.Input<Tensor>("Relu_X");
      auto relu_x_desc = helper::GetGeneralTensorDescriptor(
          dim_x, layout_format, uid, 16, tensor_format);
      data_ptrs.push_back(const_cast<T*>(relu_x_tensor->data<T>()));
      uids.push_back(uid++);
      // additional virtual outputs
      auto final_bitmask_desc = helper::GetGeneralTensorDescriptor(
          dim_x, layout_format, uid++, 16, tensor_format_math, true);

      // build ops
      auto add1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                             compute_dtype,
                                             bn1_x_desc,
                                             bn1_mean_desc,
                                             after_add1,
                                             1.0,
                                             -1.0);
      backend_ops.push_back(add1_op.get_raw_desc());

      auto mul1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                             compute_dtype,
                                             after_add1,
                                             bn1_inv_std_desc,
                                             after_mul1);
      backend_ops.push_back(mul1_op.get_raw_desc());

      auto mul2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                             compute_dtype,
                                             after_mul1,
                                             bn1_scale_desc,
                                             after_mul2);
      backend_ops.push_back(mul2_op.get_raw_desc());

      auto add2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                             compute_dtype,
                                             after_mul2,
                                             bn1_bias_desc,
                                             after_add2);
      backend_ops.push_back(add2_op.get_raw_desc());

      auto bmask_add_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                                  compute_dtype,
                                                  after_add2,
                                                  relu_x_desc,
                                                  final_bitmask_desc);
      backend_ops.push_back(bmask_add_op.get_raw_desc());

      auto drelu_op = MakeDreluOp(
          compute_dtype, *p_drelu_input_desc, final_bitmask_desc, dx_desc);
      backend_ops.push_back(drelu_op.get_raw_desc());

      auto bn_bwd_weight_op_pointer =
          BackendMakeBnbwdweightOp(compute_dtype,
                                   bn1_x_desc,
                                   bn1_mean_desc,
                                   bn1_inv_std_desc,
                                   bn1_scale_desc,
                                   dx_desc,
                                   bn1_dbeta_desc,
                                   bn1_dgamma_desc,
                                   bn1_eqscale_dy_desc,
                                   bn1_eqscale_x_desc,
                                   bn1_eqbias_desc);
      backend_ops.push_back(bn_bwd_weight_op_pointer->get_backend_descriptor());

      // make op graph
      auto op_graph = BackendBuildOperationGraph(
          handle, backend_ops.data(), backend_ops.size());
      // query and execute in backend API
      BackendExecutionPlan plan;
      plan.MakePlan(handle, 0, op_graph->get_backend_descriptor());

      auto workspace_size = plan.GetWorkspaceSize();

      helper::ExecutePlan(handle,
                          &workspace_handle,
                          &data_ptrs,
                          &uids,
                          plan.GetDesc(),
                          workspace_size);
    } else if (fuse_dual) {
      // additional inputs
      auto* bn2_mean_tensor = ctx.Input<Tensor>("BN2_mean");
      auto* bn2_inv_std_tensor = ctx.Input<Tensor>("BN2_inv_std");
      auto* bn2_scale_tensor = ctx.Input<Tensor>("BN2_scale");
      auto* bn2_bias_tensor = ctx.Input<Tensor>("BN2_bias");
      auto* bn2_x_tensor = ctx.Input<Tensor>("BN2_X");

      auto bn2_mean_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(const_cast<U*>(bn2_mean_tensor->data<U>()));
      uids.push_back(uid++);

      auto bn2_inv_std_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(const_cast<U*>(bn2_inv_std_tensor->data<U>()));
      uids.push_back(uid++);

      auto bn2_scale_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(const_cast<U*>(bn2_scale_tensor->data<U>()));
      uids.push_back(uid++);

      auto bn2_bias_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(const_cast<U*>(bn2_bias_tensor->data<U>()));
      uids.push_back(uid++);

      auto bn2_x_desc = helper::GetGeneralTensorDescriptor(
          dim_x, layout_format, uid, 16, tensor_format);
      data_ptrs.push_back(const_cast<T*>(bn2_x_tensor->data<T>()));
      uids.push_back(uid++);

      // additional virtual outputs
      auto after_dual_add1 = helper::GetGeneralTensorDescriptor(
          dim_x, layout_format, uid++, 16, tensor_format_math, true);
      auto after_dual_mul1 = helper::GetGeneralTensorDescriptor(
          dim_x, layout_format, uid++, 16, tensor_format_math, true);
      auto after_dual_add2 = helper::GetGeneralTensorDescriptor(
          dim_x, layout_format, uid++, 16, tensor_format_math, true);
      auto after_dual_mul2 = helper::GetGeneralTensorDescriptor(
          dim_x, layout_format, uid++, 16, tensor_format_math, true);
      auto final_bitmask_desc = helper::GetGeneralTensorDescriptor(
          dim_x, layout_format, uid++, 16, tensor_format_math, true);
      // additional outputs
      auto* bn2_dgamma_tensor = ctx.Output<Tensor>("BN2_dGamma");
      auto* bn2_dbeta_tensor = ctx.Output<Tensor>("BN2_dBeta");
      auto* bn2_eqscale_dy_tensor = ctx.Output<Tensor>("BN2_eqscale_dy");
      auto* bn2_eqscale_x_tensor = ctx.Output<Tensor>("BN2_eqscale_x");
      auto* bn2_eqbias_tensor = ctx.Output<Tensor>("BN2_eqbias");

      bn2_dgamma_tensor->mutable_data<U>(ctx.GetPlace());
      bn2_dbeta_tensor->mutable_data<U>(ctx.GetPlace());
      bn2_eqscale_dy_tensor->mutable_data<U>(ctx.GetPlace());
      bn2_eqscale_x_tensor->mutable_data<U>(ctx.GetPlace());
      bn2_eqbias_tensor->mutable_data<U>(ctx.GetPlace());

      auto bn2_dgamma_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(bn2_dgamma_tensor->data<U>());
      uids.push_back(uid++);

      auto bn2_dbeta_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(bn2_dbeta_tensor->data<U>());
      uids.push_back(uid++);

      auto bn2_eqscale_dy_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(bn2_eqscale_dy_tensor->data<U>());
      uids.push_back(uid++);

      auto bn2_eqscale_x_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(bn2_eqscale_x_tensor->data<U>());
      uids.push_back(uid++);

      auto bn2_eqbias_desc = helper::GetGeneralTensorDescriptor(
          dim_scale, layout_format, uid, 16, tensor_format_math);
      data_ptrs.push_back(bn2_eqbias_tensor->data<U>());
      uids.push_back(uid++);

      // build ops
      auto add1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                             compute_dtype,
                                             bn1_x_desc,
                                             bn1_mean_desc,
                                             after_add1,
                                             1.0,
                                             -1.0);
      backend_ops.push_back(add1_op.get_raw_desc());

      auto mul1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                             compute_dtype,
                                             after_add1,
                                             bn1_inv_std_desc,
                                             after_mul1);
      backend_ops.push_back(mul1_op.get_raw_desc());

      auto mul2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                             compute_dtype,
                                             after_mul1,
                                             bn1_scale_desc,
                                             after_mul2);
      backend_ops.push_back(mul2_op.get_raw_desc());

      auto add2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                             compute_dtype,
                                             after_mul2,
                                             bn1_bias_desc,
                                             after_add2);
      backend_ops.push_back(add2_op.get_raw_desc());

      auto dual_add1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                                  compute_dtype,
                                                  bn2_x_desc,
                                                  bn2_mean_desc,
                                                  after_dual_add1,
                                                  1.0,
                                                  -1.0);
      backend_ops.push_back(dual_add1_op.get_raw_desc());

      auto dual_mul1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                                  compute_dtype,
                                                  after_dual_add1,
                                                  bn2_inv_std_desc,
                                                  after_dual_mul1);
      backend_ops.push_back(dual_mul1_op.get_raw_desc());

      auto dual_mul2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                                  compute_dtype,
                                                  after_dual_mul1,
                                                  bn2_scale_desc,
                                                  after_dual_mul2);
      backend_ops.push_back(dual_mul2_op.get_raw_desc());

      auto dual_add2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                                  compute_dtype,
                                                  after_dual_mul2,
                                                  bn2_bias_desc,
                                                  after_dual_add2);
      backend_ops.push_back(dual_add2_op.get_raw_desc());

      auto bmask_add_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                                  compute_dtype,
                                                  after_add2,
                                                  after_dual_add2,
                                                  final_bitmask_desc);
      backend_ops.push_back(bmask_add_op.get_raw_desc());

      auto drelu_op = MakeDreluOp(
          compute_dtype, *p_drelu_input_desc, final_bitmask_desc, dx_desc);
      backend_ops.push_back(drelu_op.get_raw_desc());

      auto bn_bwd_weight_op_pointer =
          BackendMakeBnbwdweightOp(compute_dtype,
                                   bn1_x_desc,
                                   bn1_mean_desc,
                                   bn1_inv_std_desc,
                                   bn1_scale_desc,
                                   dx_desc,
                                   bn1_dbeta_desc,
                                   bn1_dgamma_desc,
                                   bn1_eqscale_dy_desc,
                                   bn1_eqscale_x_desc,
                                   bn1_eqbias_desc);
      backend_ops.push_back(bn_bwd_weight_op_pointer->get_backend_descriptor());

      auto dual_bn_bwd_weight_op_pointer =
          BackendMakeBnbwdweightOp(compute_dtype,
                                   bn2_x_desc,
                                   bn2_mean_desc,
                                   bn2_inv_std_desc,
                                   bn2_scale_desc,
                                   dx_desc,
                                   bn2_dbeta_desc,
                                   bn2_dgamma_desc,
                                   bn2_eqscale_dy_desc,
                                   bn2_eqscale_x_desc,
                                   bn2_eqbias_desc);
      backend_ops.push_back(
          dual_bn_bwd_weight_op_pointer->get_backend_descriptor());

      // make op graph
      auto op_graph = BackendBuildOperationGraph(
          handle, backend_ops.data(), backend_ops.size());
      // query and execute in backend API
      BackendExecutionPlan plan;
      plan.MakePlan(handle, 0, op_graph->get_backend_descriptor());

      auto workspace_size = plan.GetWorkspaceSize();

      helper::ExecutePlan(handle,
                          &workspace_handle,
                          &data_ptrs,
                          &uids,
                          plan.GetDesc(),
                          workspace_size);
    } else {
      // build ops
      auto add1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                             compute_dtype,
                                             bn1_x_desc,
                                             bn1_mean_desc,
                                             after_add1,
                                             1.0,
                                             -1.0);
      backend_ops.push_back(add1_op.get_raw_desc());

      auto mul1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                             compute_dtype,
                                             after_add1,
                                             bn1_inv_std_desc,
                                             after_mul1);
      backend_ops.push_back(mul1_op.get_raw_desc());

      auto mul2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                             compute_dtype,
                                             after_mul1,
                                             bn1_scale_desc,
                                             after_mul2);
      backend_ops.push_back(mul2_op.get_raw_desc());

      auto add2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                             compute_dtype,
                                             after_mul2,
                                             bn1_bias_desc,
                                             after_add2);
      backend_ops.push_back(add2_op.get_raw_desc());

      auto drelu_op =
          MakeDreluOp(compute_dtype, *p_drelu_input_desc, after_add2, dx_desc);
      backend_ops.push_back(drelu_op.get_raw_desc());

      auto bn_bwd_weight_op_pointer =
          BackendMakeBnbwdweightOp(compute_dtype,
                                   bn1_x_desc,
                                   bn1_mean_desc,
                                   bn1_inv_std_desc,
                                   bn1_scale_desc,
                                   dx_desc,
                                   bn1_dbeta_desc,
                                   bn1_dgamma_desc,
                                   bn1_eqscale_dy_desc,
                                   bn1_eqscale_x_desc,
                                   bn1_eqbias_desc);
      backend_ops.push_back(bn_bwd_weight_op_pointer->get_backend_descriptor());

      // make op graph
      auto op_graph = BackendBuildOperationGraph(
          handle, backend_ops.data(), backend_ops.size());
      // query and execute in backend API
      BackendExecutionPlan plan;
      plan.MakePlan(handle, 0, op_graph->get_backend_descriptor());

      auto workspace_size = plan.GetWorkspaceSize();

      helper::ExecutePlan(handle,
                          &workspace_handle,
                          &data_ptrs,
                          &uids,
                          plan.GetDesc(),
                          workspace_size);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_dgrad_drelu_bnbwdweight,
    ops::FusedDgradDreluBnBwdWeightOpKernel<paddle::platform::float16>);
