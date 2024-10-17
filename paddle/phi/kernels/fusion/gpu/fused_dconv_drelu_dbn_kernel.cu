/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"

PHI_DECLARE_bool(cudnn_deterministic);
COMMON_DECLARE_bool(cudnn_exhaustive_search);

namespace phi {
namespace fusion {

using helper = phi::CudnnFrontendConvHelper;

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;

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

template <typename T, typename Context>
void _DgradDreluBnBwdWeightImpl(const Context& dev_ctx,
                                const DenseTensor* grad_output,
                                const DenseTensor* weight,
                                const DenseTensor* bn1_mean,
                                const DenseTensor* bn1_inv_std,
                                const DenseTensor* bn1_gamma,
                                const DenseTensor* bn1_beta,
                                const DenseTensor* bn1_input,
                                const DenseTensor* residual_input,
                                const DenseTensor* bn2_mean,
                                const DenseTensor* bn2_inv_std,
                                const DenseTensor* bn2_gamma,
                                const DenseTensor* bn2_beta,
                                const DenseTensor* bn2_input,
                                const DenseTensor* grad_output_add,
                                bool fuse_shortcut,
                                bool fuse_dual,
                                bool fuse_add,
                                const std::vector<int>& strides,
                                const std::vector<int>& dilations,
                                const std::vector<int64_t>& pre_padding,
                                const std::vector<int64_t>& post_padding,
                                bool exhaustive_search,
                                bool deterministic,
                                DenseTensor* grad_conv_input,
                                DenseTensor* grad_bn1_gamma,
                                DenseTensor* grad_bn1_beta,
                                DenseTensor* bn1_coeff_a,
                                DenseTensor* bn1_coeff_b,
                                DenseTensor* bn1_coeff_c,
                                DenseTensor* grad_bn2_gamma,
                                DenseTensor* grad_bn2_beta,
                                DenseTensor* bn2_coeff_a,
                                DenseTensor* bn2_coeff_b,
                                DenseTensor* bn2_coeff_c) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kDgradDreluBnBwdWeight);
  // get handles
  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();
  // transform filter to NHWC layout
  DenseTensor w_tensor_transformed(weight->dtype());
  ResizeToChannelLast<Context, T>(dev_ctx, weight, &w_tensor_transformed);
  TransToChannelLast<Context, T>(dev_ctx, weight, &w_tensor_transformed);
  // build tensor descriptors
  cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
  auto tensor_format =
      phi::backends::gpu::ToCudnnDataType(grad_output->dtype());
  auto tensor_format_math = CUDNN_DATA_FLOAT;
  auto compute_dtype = CUDNN_DATA_FLOAT;
  // get dims in CUDNN manner: [N, C, H, W]
  auto dim_x = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(bn1_input->dims()));
  auto dim_filt = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(w_tensor_transformed.dims()));
  auto dim_y = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(grad_output->dims()));
  std::vector<int64_t> dim_scale(dim_x.size(), 1);
  dim_scale[1] = dim_x[1];  //  [1, C, 1, 1]

  std::vector<void*> data_ptrs;
  std::vector<int64_t> uids;
  int64_t uid = 100;

  // Build tensor descriptors
  // dgrad inputs
  auto dy_desc = helper::GetGeneralTensorDescriptor(
      dim_y, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(grad_output->data<T>()));
  uids.push_back(uid);

  auto w_desc = helper::GetGeneralTensorDescriptor(
      dim_filt, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(w_tensor_transformed.data<T>()));
  uids.push_back(uid);

  // dBN1 inputs
  auto bn1_mean_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(const_cast<float*>(bn1_mean->data<float>()));
  uids.push_back(uid);

  auto bn1_inv_std_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(const_cast<float*>(bn1_inv_std->data<float>()));
  uids.push_back(uid);

  auto bn1_scale_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(const_cast<float*>(bn1_gamma->data<float>()));
  uids.push_back(uid);

  auto bn1_bias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(const_cast<float*>(bn1_beta->data<float>()));
  uids.push_back(uid);

  auto bn1_x_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(bn1_input->data<T>()));
  uids.push_back(uid);

  // dBN2 inputs
  auto bn2_mean_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<float*>(bn2_mean->data<float>()));
    uids.push_back(uid);
  }

  auto bn2_inv_std_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<float*>(bn2_inv_std->data<float>()));
    uids.push_back(uid);
  }

  auto bn2_scale_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<float*>(bn2_gamma->data<float>()));
    uids.push_back(uid);
  }

  auto bn2_bias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<float*>(bn2_beta->data<float>()));
    uids.push_back(uid);
  }

  auto bn2_x_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<T*>(bn2_input->data<T>()));
    uids.push_back(uid);
  }

  // shortcut input
  auto relu_x_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  if (fuse_shortcut) {
    data_ptrs.push_back(const_cast<T*>(residual_input->data<T>()));
    uids.push_back(uid);
  }

  // fuse_add inputs
  auto dy_branch_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  if (fuse_add) {
    data_ptrs.push_back(const_cast<T*>(grad_output_add->data<T>()));
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
  data_ptrs.push_back(grad_conv_input->data<T>());
  uids.push_back(uid);

  // dBN1 outputs
  auto bn1_dgamma_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(grad_bn1_gamma->data<float>());
  uids.push_back(uid);

  auto bn1_dbeta_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(grad_bn1_beta->data<float>());
  uids.push_back(uid);

  auto bn1_eqscale_dy_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(bn1_coeff_a->data<float>());
  uids.push_back(uid);

  auto bn1_eqscale_x_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(bn1_coeff_b->data<float>());
  uids.push_back(uid);

  auto bn1_eqbias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(bn1_coeff_c->data<float>());
  uids.push_back(uid);

  // dBN2 outputs
  auto bn2_dgamma_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(grad_bn2_gamma->data<float>());
    uids.push_back(uid);
  }
  auto bn2_dbeta_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(grad_bn2_beta->data<float>());
    uids.push_back(uid);
  }
  auto bn2_eqscale_dy_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(bn2_coeff_a->data<float>());
    uids.push_back(uid);
  }
  auto bn2_eqscale_x_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(bn2_coeff_b->data<float>());
    uids.push_back(uid);
  }
  auto bn2_eqbias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format_math);
  if (fuse_dual) {
    data_ptrs.push_back(bn2_coeff_c->data<float>());
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
    plan_cache.GetPlanAndWorkspaceSize(
        feature_vector, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        cached_plan->get_raw_desc(),
                        workspace_size);
    return;
  }

  auto plans = helper::FindExecutionPlans(&op_graph,
                                          exhaustive_search,
                                          deterministic,
                                          &data_ptrs,
                                          &uids,
                                          handle,
                                          &workspace_handle);

  helper::ExecutePlansAndCache(handle,
                               &workspace_handle,
                               &data_ptrs,
                               &uids,
                               &plans,
                               exhaustive_search,
                               feature_vector,
                               &plan_cache);
}

template <typename T, typename Context>
void _DbnApplyImpl(const Context& dev_ctx,
                   const DenseTensor* dY_tensor,
                   const DenseTensor* X_tensor,
                   const DenseTensor* A_tensor,
                   const DenseTensor* B_tensor,
                   const DenseTensor* C_tensor,
                   const DenseTensor* X_dual_tensor,
                   const DenseTensor* A_dual_tensor,
                   const DenseTensor* B_dual_tensor,
                   const DenseTensor* C_dual_tensor,
                   bool fuse_dual,
                   bool exhaustive_search,
                   bool deterministic,
                   DenseTensor* dX_tensor,
                   DenseTensor* dX_dual_tensor) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kDbnApply);
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
  data_ptrs.push_back(const_cast<float*>(A_tensor->data<float>()));
  uids.push_back(uid);

  auto B_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  data_ptrs.push_back(const_cast<float*>(B_tensor->data<float>()));
  uids.push_back(uid);

  auto C_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  data_ptrs.push_back(const_cast<float*>(C_tensor->data<float>()));
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
    data_ptrs.push_back(const_cast<float*>(A_dual_tensor->data<float>()));
    uids.push_back(uid);
  }

  auto B_dual_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<float*>(B_dual_tensor->data<float>()));
    uids.push_back(uid);
  }

  auto C_dual_desc = helper::GetGeneralTensorDescriptor(
      dim_a, layout_format, ++uid, 16, tensor_format_math, false);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<float*>(C_dual_tensor->data<float>()));
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
    plan_cache.GetPlanAndWorkspaceSize(
        feature_vector, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        cached_plan->get_raw_desc(),
                        workspace_size);
    return;
  }

  auto plans = helper::FindExecutionPlans(&op_graph,
                                          exhaustive_search,
                                          deterministic,
                                          &data_ptrs,
                                          &uids,
                                          handle,
                                          &workspace_handle);

  helper::ExecutePlansAndCache(handle,
                               &workspace_handle,
                               &data_ptrs,
                               &uids,
                               &plans,
                               exhaustive_search,
                               feature_vector,
                               &plan_cache);
}

template <typename T, typename Context>
void _BnActWgradImpl(const Context& dev_ctx,
                     const DenseTensor* conv_input,
                     const DenseTensor* grad_output,
                     const DenseTensor* bn_eqscale,
                     const DenseTensor* bn_eqbias,
                     bool fuse_bn_act,
                     const std::vector<int>& strides,
                     const std::vector<int>& dilations,
                     const std::vector<int64_t>& pre_padding,
                     const std::vector<int64_t>& post_padding,
                     bool exhaustive_search,
                     bool deterministic,
                     DenseTensor* dw_tensor) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kBnActWgrad);
  // get handles
  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();
  // transform filter to NHWC layout
  DenseTensor dw_tensor_transformed(dw_tensor->dtype());
  ResizeToChannelLast<Context, T>(dev_ctx, dw_tensor, &dw_tensor_transformed);
  // create tensor descriptors
  cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
  auto tensor_format = phi::backends::gpu::ToCudnnDataType(conv_input->dtype());
  auto tensor_format_math = CUDNN_DATA_FLOAT;
  auto compute_dtype = CUDNN_DATA_FLOAT;
  // create tensor descriptors
  auto dim_x = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(conv_input->dims()));
  auto dim_filt = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(dw_tensor_transformed.dims()));
  auto dim_y = phi::backends::gpu::TransformDimOrder(
      phi::vectorize<int64_t>(grad_output->dims()));
  std::vector<int64_t> dim_scale(dim_x.size(), 1);
  dim_scale[1] = dim_x[1];  //  [1, C, 1, 1]

  std::vector<void*> data_ptrs;
  std::vector<int64_t> uids;
  int64_t uid = 100;

  // inputs
  auto x_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(conv_input->data<T>()));
  uids.push_back(uid);

  auto dy_desc = helper::GetGeneralTensorDescriptor(
      dim_y, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(grad_output->data<T>()));
  uids.push_back(uid);

  auto scale_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format);
  if (fuse_bn_act) {
    data_ptrs.push_back(const_cast<T*>(bn_eqscale->data<T>()));
    uids.push_back(uid);
  }
  auto bias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format);
  if (fuse_bn_act) {
    data_ptrs.push_back(const_cast<T*>(bn_eqbias->data<T>()));
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
    plan_cache.GetPlanAndWorkspaceSize(
        feature_vector, &cached_plan, &workspace_size, handle);
    helper::ExecutePlan(handle,
                        &workspace_handle,
                        &data_ptrs,
                        &uids,
                        cached_plan->get_raw_desc(),
                        workspace_size);
    TransToChannelFirst<Context, T>(dev_ctx, &dw_tensor_transformed, dw_tensor);
    return;
  }

  auto plans = helper::FindExecutionPlans(&op_graph,
                                          exhaustive_search,
                                          deterministic,
                                          &data_ptrs,
                                          &uids,
                                          handle,
                                          &workspace_handle);

  helper::ExecutePlansAndCache(handle,
                               &workspace_handle,
                               &data_ptrs,
                               &uids,
                               &plans,
                               exhaustive_search,
                               feature_vector,
                               &plan_cache);
  // transfer back to NCWH
  TransToChannelFirst<Context, T>(dev_ctx, &dw_tensor_transformed, dw_tensor);
}

/*
his op includes 3 kernels:
1. FusedDgradDreluBnBwdWeight
Ref:
https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#dgraddrelubnbwdweight
It fuses the backward of the following patterns:
(1)    BN -> ReLU -> Conv

(2)    BN1 -> Add -> ReLU -> Conv
       BN2 ----^       |---> (optional branch)

(3)    BN -> Add -> ReLU -> Conv
  (shortcut)--^       |---> (optional branch)

The meaning of three attributes are:
- fuse_shortcut: Whether a shortcut is added in the forward pattern, as in (2).
- fuse_dual: Whether two BN outputs are added in the forward pattern, as in (3).
- fuse_add: Whether ReLU output is used in a forward node other than Conv,
  marked in (2)(3) as (optional branch). In this case, the gradient of the
branch should be added to the output dgrad.

2. DbnApply
Ref:
https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#dualdbnapply
By default it performs the following:
dX = A* dY + B * X + C
With fuse_dual:
dX = A * dY + B * X + C
dX_dual = A_dual * dY + B_dual * X_dual + C_dual

3. ConvBnWgrad
Ref:
https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#convbnwgrad
It fuses the following pattern:

X = ReLU(BN_X * Scale + Bias)
dW = Wgrad(dY, X)

Requirements:
- All tensors should have layout NHWC, except that weight, grad_weight are NCHW.
- bn_dgamma, bn_dbeta, bn_mean, bn_inv_std, bn_scale, bn_bias should have shape
[C] and dtype FP32.
- bn1_eqscale, bn1_eqbias should shape [C] and dtype FP16.
- bn_input, grad_input, residual_input, conv_input should have input shape of
Conv and dtype FP16.
*/
template <typename T, typename Context>
void FusedDconvDreluDbnKernel(
    const Context& dev_ctx,
    const DenseTensor& grad_output,
    const DenseTensor& weight,
    const paddle::optional<DenseTensor>& grad_output_add,
    const paddle::optional<DenseTensor>& residual_input,
    const paddle::optional<DenseTensor>& bn1_eqscale,
    const paddle::optional<DenseTensor>& bn1_eqbias,
    const paddle::optional<DenseTensor>& conv_input,
    const DenseTensor& bn1_mean,
    const DenseTensor& bn1_inv_std,
    const DenseTensor& bn1_gamma,
    const DenseTensor& bn1_beta,
    const DenseTensor& bn1_input,
    const paddle::optional<DenseTensor>& bn2_mean,
    const paddle::optional<DenseTensor>& bn2_inv_std,
    const paddle::optional<DenseTensor>& bn2_gamma,
    const paddle::optional<DenseTensor>& bn2_beta,
    const paddle::optional<DenseTensor>& bn2_input,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::string& padding_algorithm,
    int groups,
    const std::string& data_format,
    bool fuse_shortcut,
    bool fuse_dual,
    bool fuse_add,
    bool exhaustive_search,
    DenseTensor* grad_weight,
    DenseTensor* grad_bn1_input,
    DenseTensor* grad_bn1_gamma,
    DenseTensor* grad_bn1_beta,
    DenseTensor* grad_bn2_input,
    DenseTensor* grad_bn2_gamma,
    DenseTensor* grad_bn2_beta) {
  PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                    80,
                    common::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        dev_ctx.GetComputeCapability()));
  auto cudnn_version = phi::backends::gpu::DnnVersion();
  PADDLE_ENFORCE_GE(cudnn_version,
                    8900,
                    common::errors::PreconditionNotMet(
                        "This op only supports CUDNN version >= 8900, "
                        "but got %d.",
                        cudnn_version));
  // Attributes
  bool fuse_wgrad_bn_act = !(fuse_shortcut || fuse_dual);
  exhaustive_search = exhaustive_search || FLAGS_cudnn_exhaustive_search;
  bool deterministic = FLAGS_cudnn_deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_search && deterministic,
                    false,
                    common::errors::InvalidArgument(
                        "Can't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));
  // update padding and dilation
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  auto in_dims = bn1_input.dims();
  auto filter_dims = weight.dims();
  DDim in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
  DDim filter_data_dims = slice_ddim(
      filter_dims, 2, filter_dims.size());  // weight is in NCHW format
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                in_data_dims,
                                strides,
                                ksize);
  int data_dim = strides.size();  // 2d or 3d
  std::vector<int64_t> pre_padding(data_dim, 0);
  std::vector<int64_t> post_padding(data_dim, 0);
  for (size_t i = 0; i < data_dim; ++i) {
    pre_padding[i] = static_cast<int64_t>(paddings_vec[2 * i]);
    post_padding[i] = static_cast<int64_t>(paddings_vec[2 * i + 1]);
  }
  // alloc output variables
  dev_ctx.template Alloc<T>(grad_weight);
  dev_ctx.template Alloc<T>(grad_bn1_input);
  dev_ctx.template Alloc<float>(grad_bn1_gamma);
  dev_ctx.template Alloc<float>(grad_bn1_beta);
  if (fuse_shortcut || fuse_dual) {
    dev_ctx.template Alloc<T>(grad_bn2_input);
  }
  if (fuse_dual) {
    dev_ctx.template Alloc<float>(grad_bn2_gamma);
    dev_ctx.template Alloc<float>(grad_bn2_beta);
  }
  // intermediate buffers
  DenseTensor grad_conv_input(bn1_input.dtype());
  if (fuse_shortcut) {
    grad_conv_input.ShareDataWith(*grad_bn2_input);
  } else {
    grad_conv_input.Resize(bn1_input.dims());
    dev_ctx.template Alloc<T>(&grad_conv_input);
  }
  auto bn_dtype = bn1_mean.dtype();
  auto bn_dims = bn1_mean.dims();
  DenseTensor bn1_coeff_a(bn_dtype);
  DenseTensor bn1_coeff_b(bn_dtype);
  DenseTensor bn1_coeff_c(bn_dtype);
  DenseTensor bn2_coeff_a(bn_dtype);
  DenseTensor bn2_coeff_b(bn_dtype);
  DenseTensor bn2_coeff_c(bn_dtype);
  bn1_coeff_a.Resize(bn_dims);
  dev_ctx.template Alloc<float>(&bn1_coeff_a);
  bn1_coeff_b.Resize(bn_dims);
  dev_ctx.template Alloc<float>(&bn1_coeff_b);
  bn1_coeff_c.Resize(bn_dims);
  dev_ctx.template Alloc<float>(&bn1_coeff_c);
  if (fuse_dual) {
    bn2_coeff_a.Resize(bn_dims);
    dev_ctx.template Alloc<float>(&bn2_coeff_a);
    bn2_coeff_b.Resize(bn_dims);
    dev_ctx.template Alloc<float>(&bn2_coeff_b);
    bn2_coeff_c.Resize(bn_dims);
    dev_ctx.template Alloc<float>(&bn2_coeff_c);
  }
  // Step 1: DgradDreluBnBwdWeight
  _DgradDreluBnBwdWeightImpl<T, Context>(dev_ctx,
                                         &grad_output,
                                         &weight,
                                         &bn1_mean,
                                         &bn1_inv_std,
                                         &bn1_gamma,
                                         &bn1_beta,
                                         &bn1_input,
                                         paddle::get_pointer(residual_input),
                                         paddle::get_pointer(bn2_mean),
                                         paddle::get_pointer(bn2_inv_std),
                                         paddle::get_pointer(bn2_gamma),
                                         paddle::get_pointer(bn2_beta),
                                         paddle::get_pointer(bn2_input),
                                         paddle::get_pointer(grad_output_add),
                                         fuse_shortcut,
                                         fuse_dual,
                                         fuse_add,
                                         strides,
                                         dilations_vec,
                                         pre_padding,
                                         post_padding,
                                         exhaustive_search,
                                         deterministic,
                                         &grad_conv_input,
                                         grad_bn1_gamma,
                                         grad_bn1_beta,
                                         &bn1_coeff_a,
                                         &bn1_coeff_b,
                                         &bn1_coeff_c,
                                         grad_bn2_gamma,
                                         grad_bn2_beta,
                                         &bn2_coeff_a,
                                         &bn2_coeff_b,
                                         &bn2_coeff_c);
  // Step 2: dBN Apply
  _DbnApplyImpl<T, Context>(dev_ctx,
                            &grad_conv_input,
                            &bn1_input,
                            &bn1_coeff_a,
                            &bn1_coeff_b,
                            &bn1_coeff_c,
                            paddle::get_pointer(bn2_input),
                            &bn2_coeff_a,
                            &bn2_coeff_b,
                            &bn2_coeff_c,
                            fuse_dual,
                            exhaustive_search,
                            deterministic,
                            grad_bn1_input,
                            grad_bn2_input);

  // Step 3: Wgrad
  const DenseTensor* wgrad_conv_input =
      fuse_wgrad_bn_act ? &bn1_input : paddle::get_pointer(conv_input);
  _BnActWgradImpl<T, Context>(dev_ctx,
                              wgrad_conv_input,
                              &grad_output,
                              paddle::get_pointer(bn1_eqscale),
                              paddle::get_pointer(bn1_eqbias),
                              fuse_wgrad_bn_act,
                              strides,
                              dilations,
                              pre_padding,
                              post_padding,
                              exhaustive_search,
                              deterministic,
                              grad_weight);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dconv_drelu_dbn,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDconvDreluDbnKernel,
                   phi::dtype::float16) {
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
}
