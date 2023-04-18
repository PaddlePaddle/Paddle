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
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using helper = phi::CudnnFrontendConvHelper;

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

namespace {
cudnn_frontend::Operation MakeDreluOp(cudnnDataType_t dtype,
                                      cudnn_frontend::Tensor const& dy_desc,
                                      cudnn_frontend::Tensor const& x_desc,
                                      cudnn_frontend::Tensor const& dx_desc) {
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

cudnn_frontend::Operation MakeBnbwdweightOp(
    cudnnDataType_t dtype,
    cudnn_frontend::Tensor const& x_desc,
    cudnn_frontend::Tensor const& mean_desc,
    cudnn_frontend::Tensor const& invstd_desc,
    cudnn_frontend::Tensor const& bn_scale_desc,
    cudnn_frontend::Tensor const& dy_desc,
    cudnn_frontend::Tensor const& dbn_bias_desc,
    cudnn_frontend::Tensor const& dbn_scale_desc,
    cudnn_frontend::Tensor const& eq_dy_scale_desc,
    cudnn_frontend::Tensor const& eq_x_scale_desc,
    cudnn_frontend::Tensor const& eqbias_desc) {
  auto op =
      cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
          .setComputeType(dtype)
          .setxDesc(x_desc)
          .setSavedMeanAndInvVar(mean_desc, invstd_desc)
          .setScale(bn_scale_desc)
          .setdyDesc(dy_desc)
          .setEqScalesAndBias(eq_dy_scale_desc, eq_x_scale_desc, eqbias_desc)
          .setDScaleAndDBias(dbn_scale_desc, dbn_bias_desc)
          .build();
  VLOG(6) << op.describe();
  return op;
}
}  // namespace

template <typename T>
void _DgradDreluBnBwdWeightImpl(const framework::ExecutionContext& ctx,
                                const Tensor* dy_tensor,
                                const Tensor* w_tensor,
                                const Tensor* bn1_mean_tensor,
                                const Tensor* bn1_inv_std_tensor,
                                const Tensor* bn1_scale_tensor,
                                const Tensor* bn1_bias_tensor,
                                const Tensor* bn1_x_tensor,
                                const Tensor* relu_x_tensor,
                                const Tensor* bn2_mean_tensor,
                                const Tensor* bn2_inv_std_tensor,
                                const Tensor* bn2_scale_tensor,
                                const Tensor* bn2_bias_tensor,
                                const Tensor* bn2_x_tensor,
                                const Tensor* dy_branch_tensor,
                                bool fuse_shortcut,
                                bool fuse_dual,
                                bool fuse_add,
                                const std::vector<int>& strides,
                                const std::vector<int>& dilations,
                                const std::vector<int64_t>& pre_padding,
                                const std::vector<int64_t>& post_padding,
                                Tensor* dx_tensor,
                                Tensor* bn1_dgamma_tensor,
                                Tensor* bn1_dbeta_tensor,
                                Tensor* bn1_eqscale_dy_tensor,
                                Tensor* bn1_eqscale_x_tensor,
                                Tensor* bn1_eqbias_tensor,
                                Tensor* bn2_dgamma_tensor,
                                Tensor* bn2_dbeta_tensor,
                                Tensor* bn2_eqscale_dy_tensor,
                                Tensor* bn2_eqscale_x_tensor,
                                Tensor* bn2_eqbias_tensor) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kDgradDreluBnBwdWeight);

  using U = BatchNormParamType<T>;
  auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
  // get handles
  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();
  // transform filter to NHWC layout
  Tensor w_tensor_transformed(w_tensor->dtype());
  using Context = phi::GPUContext;
  ResizeToChannelLast<Context, T>(ctx, w_tensor, &w_tensor_transformed);
  TransToChannelLast<Context, T>(ctx, w_tensor, &w_tensor_transformed);
  // build tensor descriptors
  cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
  auto tensor_format = phi::backends::gpu::ToCudnnDataType(dy_tensor->dtype());
  auto tensor_format_math = CUDNN_DATA_FLOAT;
  auto compute_dtype = CUDNN_DATA_FLOAT;
  // get dims in CUDNN manner: [N, C, H, W]
  auto dim_x = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(bn1_x_tensor->dims()));
  auto dim_filt = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(w_tensor_transformed.dims()));
  auto dim_y = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(dy_tensor->dims()));
  std::vector<int64_t> dim_scale(dim_x.size(), 1);
  dim_scale[1] = dim_x[1];  //  [1, C, 1, 1]

  std::vector<void*> data_ptrs;
  std::vector<int64_t> uids;
  int64_t uid = 100;

  // Build tensor descriptors
  // dgrad inputs
  auto dy_desc = helper::GetGeneralTensorDescriptor(
      dim_y, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(dy_tensor->data<T>()));
  uids.push_back(uid);

  auto w_desc = helper::GetGeneralTensorDescriptor(
      dim_filt, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(w_tensor_transformed.data<T>()));
  uids.push_back(uid);

  // dBN1 inputs
  auto bn1_mean_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(const_cast<U*>(bn1_mean_tensor->data<U>()));
  uids.push_back(uid);

  auto bn1_inv_std_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(const_cast<U*>(bn1_inv_std_tensor->data<U>()));
  uids.push_back(uid);

  auto bn1_scale_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(const_cast<U*>(bn1_scale_tensor->data<U>()));
  uids.push_back(uid);

  auto bn1_bias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(const_cast<U*>(bn1_bias_tensor->data<U>()));
  uids.push_back(uid);

  auto bn1_x_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(bn1_x_tensor->data<T>()));
  uids.push_back(uid);

  // dBN2 inputs
  auto bn2_mean_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<U*>(bn2_mean_tensor->data<U>()));
    uids.push_back(uid);
  }

  auto bn2_inv_std_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<U*>(bn2_inv_std_tensor->data<U>()));
    uids.push_back(uid);
  }

  auto bn2_scale_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<U*>(bn2_scale_tensor->data<U>()));
    uids.push_back(uid);
  }

  auto bn2_bias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<U*>(bn2_bias_tensor->data<U>()));
    uids.push_back(uid);
  }

  auto bn2_x_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<T*>(bn2_x_tensor->data<T>()));
    uids.push_back(uid);
  }

  // shortcut input
  auto relu_x_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  if (fuse_shortcut) {
    data_ptrs.push_back(const_cast<T*>(relu_x_tensor->data<T>()));
    uids.push_back(uid);
  }

  // fuse_add inputs
  auto dy_branch_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  if (fuse_add) {
    data_ptrs.push_back(const_cast<T*>(dy_branch_tensor->data<T>()));
    uids.push_back(uid);
  }

  // virtual outputs
  auto dx_dgrad_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_add0 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_add1 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_mul1 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_add2 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_mul2 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto final_bitmask_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_dual_add1 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_dual_mul1 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_dual_add2 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_dual_mul2 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  // drelu outputs
  auto dx_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(dx_tensor->data<T>());
  uids.push_back(uid);

  // dBN1 outputs
  auto bn1_dgamma_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(bn1_dgamma_tensor->data<U>());
  uids.push_back(uid);

  auto bn1_dbeta_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(bn1_dbeta_tensor->data<U>());
  uids.push_back(uid);

  auto bn1_eqscale_dy_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(bn1_eqscale_dy_tensor->data<U>());
  uids.push_back(uid);

  auto bn1_eqscale_x_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(bn1_eqscale_x_tensor->data<U>());
  uids.push_back(uid);

  auto bn1_eqbias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(bn1_eqbias_tensor->data<U>());
  uids.push_back(uid);

  // dBN2 outputs
  auto bn2_dgamma_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(bn2_dgamma_tensor->data<U>());
    uids.push_back(uid);
  }
  auto bn2_dbeta_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(bn2_dbeta_tensor->data<U>());
    uids.push_back(uid);
  }
  auto bn2_eqscale_dy_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(bn2_eqscale_dy_tensor->data<U>());
    uids.push_back(uid);
  }
  auto bn2_eqscale_x_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(bn2_eqscale_x_tensor->data<U>());
    uids.push_back(uid);
  }
  auto bn2_eqbias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(bn2_eqbias_tensor->data<U>());
    uids.push_back(uid);
  }

  // build ops
  std::vector<cudnn_frontend::Operation const*> ops;
  // make dgrad op
  std::vector<int64_t> stride_int64 = helper::GetInt64Array(strides);
  std::vector<int64_t> dilation_int64 = helper::GetInt64Array(dilations);
  int64_t data_dim = pre_padding.size();
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
  ops.push_back(&dgrad_op);

  cudnn_frontend::Tensor* p_drelu_input_desc = &dx_dgrad_desc;
  auto add0_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                         compute_dtype,
                                         dx_dgrad_desc,
                                         dy_branch_desc,
                                         after_add0);
  if (fuse_add) {
    ops.push_back(&add0_op);
    p_drelu_input_desc = &after_add0;
  }
  // make pointwise nodes
  auto add1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                         compute_dtype,
                                         bn1_x_desc,
                                         bn1_mean_desc,
                                         after_add1,
                                         1.0,
                                         -1.0);
  ops.push_back(&add1_op);

  auto mul1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                         compute_dtype,
                                         after_add1,
                                         bn1_inv_std_desc,
                                         after_mul1);
  ops.push_back(&mul1_op);

  auto mul2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                         compute_dtype,
                                         after_mul1,
                                         bn1_scale_desc,
                                         after_mul2);
  ops.push_back(&mul2_op);

  auto add2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                         compute_dtype,
                                         after_mul2,
                                         bn1_bias_desc,
                                         after_add2);
  ops.push_back(&add2_op);

  auto dual_add1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                              compute_dtype,
                                              bn2_x_desc,
                                              bn2_mean_desc,
                                              after_dual_add1,
                                              1.0,
                                              -1.0);
  if (fuse_dual) ops.push_back(&dual_add1_op);

  auto dual_mul1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                              compute_dtype,
                                              after_dual_add1,
                                              bn2_inv_std_desc,
                                              after_dual_mul1);
  if (fuse_dual) ops.push_back(&dual_mul1_op);

  auto dual_mul2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                              compute_dtype,
                                              after_dual_mul1,
                                              bn2_scale_desc,
                                              after_dual_mul2);
  if (fuse_dual) ops.push_back(&dual_mul2_op);

  auto dual_add2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                              compute_dtype,
                                              after_dual_mul2,
                                              bn2_bias_desc,
                                              after_dual_add2);
  if (fuse_dual) ops.push_back(&dual_add2_op);

  cudnn_frontend::Tensor* p_bmask_input_desc =
      fuse_shortcut ? &relu_x_desc : &after_dual_add2;
  auto bmask_add_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                              compute_dtype,
                                              after_add2,
                                              *p_bmask_input_desc,
                                              final_bitmask_desc);
  if (fuse_shortcut || fuse_dual) ops.push_back(&bmask_add_op);

  cudnn_frontend::Tensor* p_drelu_bmask_desc =
      (fuse_shortcut || fuse_dual) ? &final_bitmask_desc : &after_add2;
  auto drelu_op = MakeDreluOp(
      compute_dtype, *p_drelu_input_desc, *p_drelu_bmask_desc, dx_desc);
  ops.push_back(&drelu_op);

  auto bn_bwd_weight_op = MakeBnbwdweightOp(compute_dtype,
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
  ops.push_back(&bn_bwd_weight_op);

  auto dual_bn_bwd_weight_op = MakeBnbwdweightOp(compute_dtype,
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
  if (fuse_dual) ops.push_back(&dual_bn_bwd_weight_op);

  // build op graph
  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  VLOG(6) << op_graph.describe();

  cudnn_frontend::feature_vector_t feature_vector;
  phi::autotune::BuildFeatureVector(&feature_vector,
                                    dim_x,
                                    dim_filt,
                                    fuse_shortcut,
                                    fuse_dual,
                                    fuse_add,
                                    strides,
                                    dilations,
                                    pre_padding,
                                    post_padding);

  if (plan_cache.FindPlan(feature_vector, handle)) {
    const cudnn_frontend::ExecutionPlan* cached_plan = nullptr;
    int64_t workspace_size = 0;
    plan_cache.GetPlan(feature_vector, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        cached_plan->get_raw_desc(),
                        workspace_size);
    return;
  }

  auto plan = helper::GetPlanByHeuristics(std::move(op_graph), handle);
  VLOG(6) << "Plan tag: " << plan.getTag();

  auto workspace_size = plan.getWorkspaceSize();
  VLOG(4) << plan.describe() << " requires workspace " << workspace_size;

  helper::ExecutePlan(handle,
                      &workspace_handle,
                      &data_ptrs,
                      &uids,
                      plan.get_raw_desc(),
                      workspace_size);

  plan_cache.InsertPlan(feature_vector, plan, handle);
}

template <typename T>
void _DbnApplyImpl(const framework::ExecutionContext& ctx,
                   const Tensor* dY_tensor,
                   const Tensor* X_tensor,
                   const Tensor* A_tensor,
                   const Tensor* B_tensor,
                   const Tensor* C_tensor,
                   const Tensor* X_dual_tensor,
                   const Tensor* A_dual_tensor,
                   const Tensor* B_dual_tensor,
                   const Tensor* C_dual_tensor,
                   bool fuse_dual,
                   Tensor* dX_tensor,
                   Tensor* dX_dual_tensor) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kDbnApply);

  using U = BatchNormParamType<T>;
  auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
  // get handles
  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();
  cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
  auto tensor_format = phi::backends::gpu::ToCudnnDataType(dY_tensor->dtype());
  auto tensor_format_math = CUDNN_DATA_FLOAT;
  auto compute_dtype = CUDNN_DATA_FLOAT;
  // build tensor descriptors
  auto dim_x = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(X_tensor->dims()));
  std::vector<int64_t> dim_a(dim_x.size(), 1);
  dim_a[1] = dim_x[1];  //  [1, C, 1, 1]

  std::vector<void*> data_ptrs;
  std::vector<int64_t> uids;
  int64_t uid = 100;

  // inputs
  auto dY_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format, false);
  data_ptrs.push_back(const_cast<T*>(dY_tensor->data<T>()));
  uids.push_back(uid);

  auto X_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format, false);
  data_ptrs.push_back(const_cast<T*>(X_tensor->data<T>()));
  uids.push_back(uid);

  auto A_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  data_ptrs.push_back(const_cast<U*>(A_tensor->data<U>()));
  uids.push_back(uid);

  auto B_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  data_ptrs.push_back(const_cast<U*>(B_tensor->data<U>()));
  uids.push_back(uid);

  auto C_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  data_ptrs.push_back(const_cast<U*>(C_tensor->data<U>()));
  uids.push_back(uid);

  auto X_dual_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format, false);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<T*>(X_dual_tensor->data<T>()));
    uids.push_back(uid);
  }

  auto A_dual_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<U*>(A_dual_tensor->data<U>()));
    uids.push_back(uid);
  }

  auto B_dual_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<U*>(B_dual_tensor->data<U>()));
    uids.push_back(uid);
  }

  auto C_dual_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<U*>(C_dual_tensor->data<U>()));
    uids.push_back(uid);
  }

  // virtual outputs
  auto after_mul0 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_mul1 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_add = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  auto after_mul0_dual = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_mul1_dual = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_add_dual = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  // outputs
  auto dX_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format, false);
  data_ptrs.push_back(dX_tensor->data<T>());
  uids.push_back(uid);

  auto dX_dual_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format, false);
  if (fuse_dual) {
    data_ptrs.push_back(dX_dual_tensor->data<T>());
    uids.push_back(uid);
  }

  // op desc
  auto mul0_op = helper::MakePointwiseOp(
      CUDNN_POINTWISE_MUL, compute_dtype, A_desc, dY_desc, after_mul0);

  auto mul1_op = helper::MakePointwiseOp(
      CUDNN_POINTWISE_MUL, compute_dtype, B_desc, X_desc, after_mul1);

  auto add0_op = helper::MakePointwiseOp(
      CUDNN_POINTWISE_ADD, compute_dtype, after_mul0, after_mul1, after_add);

  auto add1_op = helper::MakePointwiseOp(
      CUDNN_POINTWISE_ADD, compute_dtype, after_add, C_desc, dX_desc);

  auto mul0_op_dual = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                              compute_dtype,
                                              A_dual_desc,
                                              dY_desc,
                                              after_mul0_dual);

  auto mul1_op_dual = helper::MakePointwiseOp(CUDNN_POINTWISE_MUL,
                                              compute_dtype,
                                              B_dual_desc,
                                              X_dual_desc,
                                              after_mul1_dual);

  auto add0_op_dual = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                              compute_dtype,
                                              after_mul0_dual,
                                              after_mul1_dual,
                                              after_add_dual);

  auto add1_op_dual = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                              compute_dtype,
                                              after_add_dual,
                                              C_dual_desc,
                                              dX_dual_desc);

  // build op graph
  std::vector<cudnn_frontend::Operation const*> ops = {
      &mul0_op, &mul1_op, &add0_op, &add1_op};
  if (fuse_dual) {
    ops.insert(ops.end(),
               {&mul0_op_dual, &mul1_op_dual, &add0_op_dual, &add1_op_dual});
  }

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  VLOG(6) << op_graph.describe();

  cudnn_frontend::feature_vector_t feature_vector;
  phi::autotune::BuildFeatureVector(&feature_vector, dim_x, fuse_dual);

  if (plan_cache.FindPlan(feature_vector, handle)) {
    const cudnn_frontend::ExecutionPlan* cached_plan = nullptr;
    int64_t workspace_size = 0;
    plan_cache.GetPlan(feature_vector, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        cached_plan->get_raw_desc(),
                        workspace_size);
    return;
  }

  auto plan = helper::GetPlanByHeuristics(std::move(op_graph), handle);
  auto workspace_size = plan.getWorkspaceSize();
  VLOG(4) << plan.describe() << " requires workspace " << workspace_size;

  helper::ExecutePlan(handle,
                      &workspace_handle,
                      &data_ptrs,
                      &uids,
                      plan.get_raw_desc(),
                      workspace_size);

  plan_cache.InsertPlan(feature_vector, plan, handle);
}

template <typename T>
void _BnActWgradImpl(const framework::ExecutionContext& ctx,
                     const Tensor* x_tensor,
                     const Tensor* dy_tensor,
                     const Tensor* scale_tensor,
                     const Tensor* bias_tensor,
                     bool fuse_bn_act,
                     const std::vector<int>& strides,
                     const std::vector<int>& dilations,
                     const std::vector<int64_t>& pre_padding,
                     const std::vector<int64_t>& post_padding,
                     Tensor* dw_tensor) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kBnActWgrad);

  using U = BatchNormParamType<T>;
  auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
  // get handles
  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();
  // transform filter to NHWC layout
  Tensor dw_tensor_transformed(dw_tensor->dtype());
  using Context = phi::GPUContext;
  ResizeToChannelLast<Context, T>(ctx, dw_tensor, &dw_tensor_transformed);
  // create tensor descriptors
  cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
  auto tensor_format = phi::backends::gpu::ToCudnnDataType(x_tensor->dtype());
  auto tensor_format_math = CUDNN_DATA_FLOAT;
  auto compute_dtype = CUDNN_DATA_FLOAT;
  // create tensor discriptors
  auto dim_x = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(x_tensor->dims()));
  auto dim_filt = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(dw_tensor_transformed.dims()));
  auto dim_y = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(dy_tensor->dims()));
  std::vector<int64_t> dim_scale(dim_x.size(), 1);
  dim_scale[1] = dim_x[1];  //  [1, C, 1, 1]

  std::vector<void*> data_ptrs;
  std::vector<int64_t> uids;
  int64_t uid = 100;

  // inputs
  auto x_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(x_tensor->data<T>()));
  uids.push_back(uid);

  auto dy_desc = helper::GetGeneralTensorDescriptor(
      dim_y, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(dy_tensor->data<T>()));
  uids.push_back(uid);

  auto scale_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format);
  if (fuse_bn_act) {
    data_ptrs.push_back(const_cast<T*>(scale_tensor->data<T>()));
    uids.push_back(uid);
  }
  auto bias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format);
  if (fuse_bn_act) {
    data_ptrs.push_back(const_cast<T*>(bias_tensor->data<T>()));
    uids.push_back(uid);
  }

  // outputs
  auto dw_desc = helper::GetGeneralTensorDescriptor(
      dim_filt, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(dw_tensor_transformed.data<T>());
  uids.push_back(uid);

  // virtual outputs
  auto after_scale = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  auto after_bias = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  auto after_relu = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  // build ops
  std::vector<int64_t> stride_int64 = helper::GetInt64Array(strides);
  std::vector<int64_t> dilation_int64 = helper::GetInt64Array(dilations);
  int64_t data_dim = pre_padding.size();
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

  cudnn_frontend::Tensor* p_wgrad_x_desc = fuse_bn_act ? &after_relu : &x_desc;
  auto wgrad_op =
      cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
          .setdyDesc(dy_desc)
          .setdwDesc(dw_desc)
          .setxDesc(*p_wgrad_x_desc)
          .setcDesc(conv_desc)
          .setAlpha(1.0f)
          .setBeta(0.0f)
          .build();
  VLOG(6) << wgrad_op.describe();

  auto scale_op = helper::MakePointwiseOp(
      CUDNN_POINTWISE_MUL, compute_dtype, x_desc, scale_desc, after_scale);

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
  std::vector<cudnn_frontend::Operation const*> ops;
  if (fuse_bn_act)
    ops = {&wgrad_op, &scale_op, &bias_op, &relu_op};
  else
    ops = {&wgrad_op};

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  VLOG(6) << op_graph.describe();

  cudnn_frontend::feature_vector_t feature_vector;
  phi::autotune::BuildFeatureVector(&feature_vector,
                                    dim_x,
                                    dim_filt,
                                    fuse_bn_act,
                                    strides,
                                    dilations,
                                    pre_padding,
                                    post_padding);

  if (plan_cache.FindPlan(feature_vector, handle)) {
    const cudnn_frontend::ExecutionPlan* cached_plan = nullptr;
    int64_t workspace_size = 0;
    plan_cache.GetPlan(feature_vector, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        cached_plan->get_raw_desc(),
                        workspace_size);
    TransToChannelFirst<Context, T>(ctx, &dw_tensor_transformed, dw_tensor);
    return;
  }

  auto plan = helper::GetPlanByHeuristics(std::move(op_graph), handle);
  auto workspace_size = plan.getWorkspaceSize();
  VLOG(4) << plan.describe() << " requires workspace " << workspace_size;

  helper::ExecutePlan(handle,
                      &workspace_handle,
                      &data_ptrs,
                      &uids,
                      plan.get_raw_desc(),
                      workspace_size);

  plan_cache.InsertPlan(feature_vector, plan, handle);

  // transfer back to NCWH
  TransToChannelFirst<Context, T>(ctx, &dw_tensor_transformed, dw_tensor);
}

template <typename T>
class FusedDconvDreluDbnOpKernel : public framework::OpKernel<T> {
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
    PADDLE_ENFORCE_GE(cudnn_version,
                      8900,
                      phi::errors::PreconditionNotMet(
                          "This op only supports CUDNN version >= 8800, "
                          "but got %d.",
                          cudnn_version));
    // Attributes
    bool fuse_shortcut = ctx.Attr<bool>("fuse_shortcut");
    bool fuse_dual = ctx.Attr<bool>("fuse_dual");
    bool fuse_add = ctx.Attr<bool>("fuse_add");
    bool fuse_wgrad_bn_act = !(fuse_shortcut || fuse_dual);
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    // required input variables
    const Tensor* dy_tensor = ctx.Input<Tensor>("dY");
    const Tensor* w_tensor = ctx.Input<Tensor>("W");
    const Tensor* bn1_mean_tensor = ctx.Input<Tensor>("BN1_mean");
    const Tensor* bn1_inv_std_tensor = ctx.Input<Tensor>("BN1_inv_std");
    const Tensor* bn1_scale_tensor = ctx.Input<Tensor>("BN1_scale");
    const Tensor* bn1_bias_tensor = ctx.Input<Tensor>("BN1_bias");
    const Tensor* bn1_x_tensor = ctx.Input<Tensor>("BN1_X");
    // dispensable inputs
    const Tensor* relu_x_tensor = nullptr;
    const Tensor* bn1_wgrad_eqscale_tensor = nullptr;
    const Tensor* bn1_wgrad_eqbias_tensor = nullptr;
    const Tensor* conv_x_tensor = nullptr;
    const Tensor* bn2_mean_tensor = nullptr;
    const Tensor* bn2_inv_std_tensor = nullptr;
    const Tensor* bn2_scale_tensor = nullptr;
    const Tensor* bn2_bias_tensor = nullptr;
    const Tensor* bn2_x_tensor = nullptr;
    const Tensor* dy_branch_tensor = nullptr;
    if (fuse_shortcut) {
      relu_x_tensor = ctx.Input<Tensor>("Relu_X");
    }
    if (fuse_dual) {
      bn2_mean_tensor = ctx.Input<Tensor>("BN2_mean");
      bn2_inv_std_tensor = ctx.Input<Tensor>("BN2_inv_std");
      bn2_scale_tensor = ctx.Input<Tensor>("BN2_scale");
      bn2_bias_tensor = ctx.Input<Tensor>("BN2_bias");
      bn2_x_tensor = ctx.Input<Tensor>("BN2_X");
    }
    if (fuse_add) {
      dy_branch_tensor = ctx.Input<Tensor>("dY_branch");
    }
    if (!fuse_wgrad_bn_act) {
      conv_x_tensor = ctx.Input<Tensor>("Conv_X");
    } else {
      bn1_wgrad_eqscale_tensor = ctx.Input<Tensor>("BN1_eqscale");
      bn1_wgrad_eqbias_tensor = ctx.Input<Tensor>("BN1_eqbias");
    }
    // required output variables
    Tensor* bn1_dx_tensor = ctx.Output<Tensor>("BN1_dX");
    Tensor* bn1_dgamma_tensor = ctx.Output<Tensor>("BN1_dGamma");
    Tensor* bn1_dbeta_tensor = ctx.Output<Tensor>("BN1_dBeta");
    Tensor* dw_tensor = ctx.Output<Tensor>("dW");
    bn1_dx_tensor->mutable_data<T>(ctx.GetPlace());
    bn1_dgamma_tensor->mutable_data<U>(ctx.GetPlace());
    bn1_dbeta_tensor->mutable_data<U>(ctx.GetPlace());
    dw_tensor->mutable_data<T>(ctx.GetPlace());
    // dispensable outputs
    Tensor* bn2_dx_tensor = nullptr;
    Tensor* bn2_dgamma_tensor = nullptr;
    Tensor* bn2_dbeta_tensor = nullptr;
    if (fuse_shortcut || fuse_dual) {
      bn2_dx_tensor = ctx.Output<Tensor>("BN2_dX");
      bn2_dx_tensor->mutable_data<T>(ctx.GetPlace());
    }
    if (fuse_dual) {
      bn2_dgamma_tensor = ctx.Output<Tensor>("BN2_dGamma");
      bn2_dbeta_tensor = ctx.Output<Tensor>("BN2_dBeta");
      bn2_dgamma_tensor->mutable_data<U>(ctx.GetPlace());
      bn2_dbeta_tensor->mutable_data<U>(ctx.GetPlace());
    }
    // update padding and dilation
    auto in_dims = bn1_x_tensor->dims();
    auto filter_dims = w_tensor->dims();
    framework::DDim in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
    framework::DDim filter_data_dims = slice_ddim(
        filter_dims, 2, filter_dims.size());  // w_tensor is in NCHW format
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
    // intermediate buffers
    Tensor dx_tensor(bn1_x_tensor->dtype());
    if (fuse_shortcut) {
      dx_tensor.ShareDataWith(*bn2_dx_tensor);
    } else {
      dx_tensor.Resize(bn1_x_tensor->dims());
      dx_tensor.mutable_data<T>(ctx.GetPlace());
    }
    auto bn_dtype = bn1_mean_tensor->dtype();
    auto bn_dims = bn1_mean_tensor->dims();
    Tensor bn1_eqscale_dy_tensor(bn_dtype);
    Tensor bn1_eqscale_x_tensor(bn_dtype);
    Tensor bn1_eqbias_tensor(bn_dtype);
    Tensor bn2_eqscale_dy_tensor(bn_dtype);
    Tensor bn2_eqscale_x_tensor(bn_dtype);
    Tensor bn2_eqbias_tensor(bn_dtype);
    bn1_eqscale_dy_tensor.Resize(bn_dims);
    bn1_eqscale_dy_tensor.mutable_data<U>(ctx.GetPlace());
    bn1_eqscale_x_tensor.Resize(bn_dims);
    bn1_eqscale_x_tensor.mutable_data<U>(ctx.GetPlace());
    bn1_eqbias_tensor.Resize(bn_dims);
    bn1_eqbias_tensor.mutable_data<U>(ctx.GetPlace());
    if (fuse_dual) {
      bn2_eqscale_dy_tensor.Resize(bn_dims);
      bn2_eqscale_dy_tensor.mutable_data<U>(ctx.GetPlace());
      bn2_eqscale_x_tensor.Resize(bn_dims);
      bn2_eqscale_x_tensor.mutable_data<U>(ctx.GetPlace());
      bn2_eqbias_tensor.Resize(bn_dims);
      bn2_eqbias_tensor.mutable_data<U>(ctx.GetPlace());
    }
    // Step 1: DgradDreluBnBwdWeight
    _DgradDreluBnBwdWeightImpl<T>(ctx,
                                  dy_tensor,
                                  w_tensor,
                                  bn1_mean_tensor,
                                  bn1_inv_std_tensor,
                                  bn1_scale_tensor,
                                  bn1_bias_tensor,
                                  bn1_x_tensor,
                                  relu_x_tensor,
                                  bn2_mean_tensor,
                                  bn2_inv_std_tensor,
                                  bn2_scale_tensor,
                                  bn2_bias_tensor,
                                  bn2_x_tensor,
                                  dy_branch_tensor,
                                  fuse_shortcut,
                                  fuse_dual,
                                  fuse_add,
                                  strides,
                                  dilations,
                                  pre_padding,
                                  post_padding,
                                  &dx_tensor,
                                  bn1_dgamma_tensor,
                                  bn1_dbeta_tensor,
                                  &bn1_eqscale_dy_tensor,
                                  &bn1_eqscale_x_tensor,
                                  &bn1_eqbias_tensor,
                                  bn2_dgamma_tensor,
                                  bn2_dbeta_tensor,
                                  &bn2_eqscale_dy_tensor,
                                  &bn2_eqscale_x_tensor,
                                  &bn2_eqbias_tensor);
    // Step 2: dBN Apply
    _DbnApplyImpl<T>(ctx,
                     &dx_tensor,
                     bn1_x_tensor,
                     &bn1_eqscale_dy_tensor,
                     &bn1_eqscale_x_tensor,
                     &bn1_eqbias_tensor,
                     bn2_x_tensor,
                     &bn2_eqscale_dy_tensor,
                     &bn2_eqscale_x_tensor,
                     &bn2_eqbias_tensor,
                     fuse_dual,
                     bn1_dx_tensor,
                     bn2_dx_tensor);

    // Step 3: Wgrad
    const Tensor* wgrad_x_tensor =
        fuse_wgrad_bn_act ? bn1_x_tensor : conv_x_tensor;
    _BnActWgradImpl<T>(ctx,
                       wgrad_x_tensor,
                       dy_tensor,
                       bn1_wgrad_eqscale_tensor,
                       bn1_wgrad_eqbias_tensor,
                       fuse_wgrad_bn_act,
                       strides,
                       dilations,
                       pre_padding,
                       post_padding,
                       dw_tensor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fused_dconv_drelu_dbn,
    ops::FusedDconvDreluDbnOpKernel<paddle::platform::float16>);
