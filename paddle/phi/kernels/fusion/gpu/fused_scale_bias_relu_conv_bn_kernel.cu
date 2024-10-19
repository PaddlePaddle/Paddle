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

#include <float.h>
#include <array>

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

/*
 * Implements Scale + Bias + ReLU + Conv + BNStats fusion pattern.
 * Same as the following (x and output are in NHWC format):
 * ```
 *   output = conv2d(relu(x * scale + bias), w)
 *   sum_output, sqsum_output = bnstats(output)
 * ```
 * Here, bnstats generates per-channel statistics, same as:
 * ```
 *   sum_output = output.sum(axis=[0,1,2])
 *   sqsum_output = (output ** 2).sum(axis=[0,1,2])
 * ```
 * More details:
 * https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#genstats-runtime-fusion-engine
 */
template <typename T, typename Context>
void FusedScaleBiasReluConvBnstatsImpl(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& w,
    const paddle::optional<DenseTensor>& scale,
    const paddle::optional<DenseTensor>& bias,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    const std::vector<int>& strides,
    const std::string& padding_algorithm,
    bool fuse_prologue,
    bool exhaustive_search,
    bool deterministic,
    DenseTensor* output,
    DenseTensor* sum_output,
    DenseTensor* sqsum_output) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kScaleBiasReluConvBNstats);

  // transformed tensor
  DenseTensor w_transformed(w.dtype());
  // Assume input and output already in NHWC.
  // No transformation is needed for them.
  VLOG(3) << "Transform filter tensor from NCHW to NHWC.";
  ResizeToChannelLast<Context, T>(dev_ctx, &w, &w_transformed);
  TransToChannelLast<Context, T>(dev_ctx, &w, &w_transformed);

  // update padding and dilation
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  auto in_dims = x.dims();
  auto filter_dims = w_transformed.dims();
  DDim in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
  DDim filter_data_dims = slice_ddim(filter_dims, 1, filter_dims.size() - 1);
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                in_data_dims,
                                strides,
                                ksize);

  int data_dim = strides.size();  // 2d only

  std::vector<int64_t> pre_padding(data_dim, 0);
  std::vector<int64_t> post_padding(data_dim, 0);
  for (size_t i = 0; i < data_dim; ++i) {
    pre_padding[i] = static_cast<int64_t>(paddings_vec[2 * i]);
    post_padding[i] = static_cast<int64_t>(paddings_vec[2 * i + 1]);
  }

  // input pointers
  T* input_data = const_cast<T*>(x.data<T>());
  T* filter_data = w_transformed.data<T>();

  // output pointers
  T* output_data = output->data<T>();
  float* sum_output_data = sum_output->data<float>();
  float* sqsum_output_data = sqsum_output->data<float>();

  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();

  // build tensors
  cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
  auto tensor_format = phi::backends::gpu::ToCudnnDataType(x.dtype());

  auto tensor_format_math = CUDNN_DATA_FLOAT;
  auto compute_dtype = CUDNN_DATA_FLOAT;

  // get dims in CUDNN manner: [N, C, H, W]
  auto dim_x = phi::backends::gpu::TransformDimOrder(
      common::vectorize<int64_t>(in_dims));
  auto dim_filt = phi::backends::gpu::TransformDimOrder(
      common::vectorize<int64_t>(filter_dims));
  auto dim_y = phi::backends::gpu::TransformDimOrder(
      common::vectorize<int64_t>(output->dims()));
  std::vector<int64_t> dim_scale(dim_x.size(), 1);
  dim_scale[1] = dim_x[1];                        //  [1, C, 1, 1]
  std::vector<int64_t> dim_sum(dim_x.size(), 1);  // [1, K, 1, 1]
  dim_sum[1] = dim_filt[0];

  std::vector<void*> data_ptrs;
  std::vector<int64_t> uids;
  int64_t uid = 100;

  // inputs
  auto input_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(input_data);
  uids.push_back(uid);

  auto filter_desc = helper::GetGeneralTensorDescriptor(
      dim_filt, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(filter_data);
  uids.push_back(uid);

  // dispensable inputs
  auto scale_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format);
  if (fuse_prologue) {
    data_ptrs.push_back(const_cast<T*>(scale->data<T>()));
    uids.push_back(uid);
  }

  auto bias_desc = helper::GetGeneralTensorDescriptor(
      dim_scale, layout_format, ++uid, 16, tensor_format);
  if (fuse_prologue) {
    data_ptrs.push_back(const_cast<T*>(bias->data<T>()));
    uids.push_back(uid);
  }

  // outputs
  auto output_desc = helper::GetGeneralTensorDescriptor(
      dim_y, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(output_data);
  uids.push_back(uid);

  auto sum_output_desc = helper::GetGeneralTensorDescriptor(
      dim_sum, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(sum_output_data);
  uids.push_back(uid);

  auto sqsum_output_desc = helper::GetGeneralTensorDescriptor(
      dim_sum, layout_format, ++uid, 16, tensor_format_math);
  data_ptrs.push_back(sqsum_output_data);
  uids.push_back(uid);

  // virtual outputs
  auto after_scale = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_bias = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);
  auto after_relu = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  // create ops
  auto scale_op = helper::MakePointwiseOp(
      CUDNN_POINTWISE_MUL, compute_dtype, input_desc, scale_desc, after_scale);

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
  std::vector<int64_t> dilation_int64 = helper::GetInt64Array(dilations_vec);
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
  cudnn_frontend::Tensor* input_to_conv =
      fuse_prologue ? &after_relu : &input_desc;
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
    ops =
        std::vector<cudnn_frontend::Operation const*>({&conv_op, &genstat_op});
  }

  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  VLOG(6) << op_graph.describe();

  cudnn_frontend::feature_vector_t feature_vector;
  phi::autotune::BuildFeatureVector(&feature_vector,
                                    dim_x,
                                    dim_filt,
                                    strides,
                                    paddings,
                                    dilations,
                                    pre_padding,
                                    post_padding,
                                    fuse_prologue);

  helper::QueryCacheAndExecute(handle,
                               &workspace_handle,
                               &op_graph,
                               &data_ptrs,
                               &uids,
                               exhaustive_search,
                               deterministic,
                               feature_vector,
                               &plan_cache);
}

/*
 * Implements BNFinalize pattern. It works with aforementioned bnstats node:
 * ```
 *   y = bn_finalize(genstats(conv_out))
 * ```
 * is the same as:
 * ```
 *   y = batchnorm2d(conv_out)
 * ```
 */
template <typename T, typename Context>
void BNFinalizeImpl(const Context& dev_ctx,
                    const DenseTensor& sum_tensor,
                    const DenseTensor& sqsum_tensor,
                    const DenseTensor& bn_scale,
                    const DenseTensor& bn_bias,
                    const DenseTensor& input_running_mean,
                    const DenseTensor& input_running_var,
                    int64_t accumulation_count,
                    float exp_decay,
                    float epsilon,
                    bool exhaustive_search,
                    bool deterministic,
                    DenseTensor* out_running_mean,
                    DenseTensor* out_running_var,
                    DenseTensor* saved_mean,
                    DenseTensor* saved_var,
                    DenseTensor* eq_scale,
                    DenseTensor* eq_bias) {
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kBNFinalize);

  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();
  // set dtypes
  cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
  auto tensor_format_bn =
      phi::backends::gpu::ToCudnnDataType(sum_tensor.dtype());
  auto tensor_format = phi::backends::gpu::ToCudnnDataType(eq_scale->dtype());
  auto compute_dtype = CUDNN_DATA_FLOAT;
  // create tensor descriptors
  auto dim_input = common::vectorize<int64_t>(sum_tensor.dims());
  std::vector<int64_t> dim_c = {1, dim_input[0], 1, 1};  //  [1, C, 1, 1]
  std::vector<int64_t> dim_scalar = {1, 1, 1, 1};
  std::vector<int64_t> stride_scalar = {1, 1, 1, 1};

  std::vector<void*> data_ptrs;
  std::vector<int64_t> uids;
  int64_t uid = 100;

  // inputs
  auto sum_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(const_cast<float*>(sum_tensor.data<float>()));
  uids.push_back(uid);

  auto sqsum_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(const_cast<float*>(sqsum_tensor.data<float>()));
  uids.push_back(uid);

  auto scale_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(const_cast<float*>(bn_scale.data<float>()));
  uids.push_back(uid);

  auto bias_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(const_cast<float*>(bn_bias.data<float>()));
  uids.push_back(uid);

  auto input_running_mean_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(const_cast<float*>(input_running_mean.data<float>()));
  uids.push_back(uid);

  auto input_running_var_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(const_cast<float*>(input_running_var.data<float>()));
  uids.push_back(uid);

  // outputs
  auto updated_running_mean_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(out_running_mean->data<float>());
  uids.push_back(uid);

  auto updated_running_var_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(out_running_var->data<float>());
  uids.push_back(uid);

  auto saved_mean_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(saved_mean->data<float>());
  uids.push_back(uid);

  auto saved_inv_var_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format_bn);
  data_ptrs.push_back(saved_var->data<float>());
  uids.push_back(uid);

  auto eq_scale_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(eq_scale->data<T>());
  uids.push_back(uid);

  auto eq_bias_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(eq_bias->data<T>());
  uids.push_back(uid);

  // scalar descriptors
  auto epsilon_desc = cudnn_frontend::TensorBuilder()
                          .setDim(dim_scalar.size(), dim_scalar.data())
                          .setStride(stride_scalar.size(), stride_scalar.data())
                          .setId(++uid)
                          .setAlignment(16)
                          .setDataType(CUDNN_DATA_FLOAT)
                          .setByValue(true)
                          .build();
  data_ptrs.push_back(&epsilon);
  uids.push_back(uid);

  auto exp_decay_desc =
      cudnn_frontend::TensorBuilder()
          .setDim(dim_scalar.size(), dim_scalar.data())
          .setStride(stride_scalar.size(), stride_scalar.data())
          .setId(++uid)
          .setAlignment(16)
          .setDataType(CUDNN_DATA_FLOAT)
          .setByValue(true)
          .build();
  data_ptrs.push_back(&exp_decay);
  uids.push_back(uid);

  auto accum_count_desc =
      cudnn_frontend::TensorBuilder()
          .setDim(dim_scalar.size(), dim_scalar.data())
          .setStride(stride_scalar.size(), stride_scalar.data())
          .setId(++uid)
          .setAlignment(16)
          .setDataType(CUDNN_DATA_INT64)
          .setByValue(true)
          .build();
  data_ptrs.push_back(&accumulation_count);
  uids.push_back(uid);

  //  build ops
  auto finalize_stat_op =
      cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
          .setComputeType(compute_dtype)
          .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
          .setSumDesc(sum_desc)
          .setSqSumDesc(sqsum_desc)
          .setScaleAndBias(scale_desc, bias_desc)
          .setEqScaleAndBias(eq_scale_desc, eq_bias_desc)
          .setPrevRunningMeanAndVar(input_running_mean_desc,
                                    input_running_var_desc)
          .setNextRunningMeanAndVar(updated_running_mean_desc,
                                    updated_running_var_desc)
          .setSavedMeanAndInvVar(saved_mean_desc, saved_inv_var_desc)
          .setEpsilonTensor(epsilon_desc)
          .setAccumCountTensor(accum_count_desc)
          .setExpDecayFactorTensor(exp_decay_desc)
          .build();

  std::array<cudnn_frontend::Operation const*, 1> ops = {&finalize_stat_op};
  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  VLOG(6) << op_graph.describe();

  cudnn_frontend::feature_vector_t feature_vector;
  phi::autotune::BuildFeatureVector(
      &feature_vector, dim_input, accumulation_count, exp_decay, epsilon);

  helper::QueryCacheAndExecute(handle,
                               &workspace_handle,
                               &op_graph,
                               &data_ptrs,
                               &uids,
                               exhaustive_search,
                               deterministic,
                               feature_vector,
                               &plan_cache);
}

template <typename T, typename Context>
void FusedScaleBiasReluConvBnKernel(const Context& dev_ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& w,
                                    const paddle::optional<DenseTensor>& scale,
                                    const paddle::optional<DenseTensor>& bias,
                                    const DenseTensor& bn_scale,
                                    const DenseTensor& bn_bias,
                                    const DenseTensor& input_running_mean,
                                    const DenseTensor& input_running_var,
                                    const std::vector<int>& paddings,
                                    const std::vector<int>& dilations,
                                    const std::vector<int>& strides,
                                    const std::string& padding_algorithm,
                                    int groups,
                                    const std::string& data_format,
                                    float momentum,
                                    float epsilon,
                                    bool fuse_prologue,
                                    bool exhaustive_search,
                                    int64_t accumulation_count,
                                    DenseTensor* out,
                                    DenseTensor* out_running_mean,
                                    DenseTensor* out_running_var,
                                    DenseTensor* saved_mean,
                                    DenseTensor* saved_var,
                                    DenseTensor* eq_scale,
                                    DenseTensor* eq_bias) {
  auto cudnn_version = phi::backends::gpu::DnnVersion();
  PADDLE_ENFORCE_GE(cudnn_version,
                    8800,
                    common::errors::PreconditionNotMet(
                        "This op only supports CUDNN version >= 8800, "
                        "but got %d.",
                        cudnn_version));
  PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                    80,
                    common::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        dev_ctx.GetComputeCapability()));
  // attr
  float exp_decay = 1. - momentum;
  if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
    LOG(ERROR) << "Provided epsilon is smaller than "
               << "CUDNN_BN_MIN_EPSILON. Setting it to "
               << "CUDNN_BN_MIN_EPSILON instead.";
  }
  epsilon =
      std::max(epsilon, static_cast<float>(CUDNN_BN_MIN_EPSILON + FLT_EPSILON));
  // exhaustive search
  exhaustive_search = exhaustive_search || FLAGS_cudnn_exhaustive_search;
  bool deterministic = FLAGS_cudnn_deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_search && deterministic,
                    false,
                    common::errors::InvalidArgument(
                        "Can't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));
  // check optional inputs
  if (fuse_prologue) {
    PADDLE_ENFORCE_EQ(
        scale && bias,
        true,
        common::errors::InvalidArgument(
            "\"scale\" and \"bias\" must be provided "
            "when fuse_prologue = true. Got scale = %d; bias = %d.",
            scale,
            bias));
  }

  // alloc output variables
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<float>(out_running_mean);
  dev_ctx.template Alloc<float>(out_running_var);
  dev_ctx.template Alloc<float>(saved_mean);
  dev_ctx.template Alloc<float>(saved_var);
  dev_ctx.template Alloc<T>(eq_scale);
  dev_ctx.template Alloc<T>(eq_bias);

  // deal with strides, dilations and paddings
  if (accumulation_count == 0) {
    // dim_out = [N, H, W, C]
    // accumulation_count = N * H * W
    auto dim_out = common::vectorize<int64_t>(out->dims());
    accumulation_count = dim_out[0] * dim_out[1] * dim_out[2];
  }

  // Step 1: Scale Bias ReLU Conv BNStats
  auto bn_dims = bn_scale.dims();
  DenseTensor sum_tensor(bn_scale.dtype());
  DenseTensor sqsum_tensor(bn_scale.dtype());
  sum_tensor.Resize(bn_dims);
  sqsum_tensor.Resize(bn_dims);
  dev_ctx.template Alloc<float>(&sum_tensor);
  dev_ctx.template Alloc<float>(&sqsum_tensor);
  FusedScaleBiasReluConvBnstatsImpl<T, Context>(dev_ctx,
                                                x,
                                                w,
                                                scale,
                                                bias,
                                                paddings,
                                                dilations,
                                                strides,
                                                padding_algorithm,
                                                fuse_prologue,
                                                exhaustive_search,
                                                deterministic,
                                                out,
                                                &sum_tensor,
                                                &sqsum_tensor);
  // Step 2: BN Finalize
  BNFinalizeImpl<T, Context>(dev_ctx,
                             sum_tensor,
                             sqsum_tensor,
                             bn_scale,
                             bn_bias,
                             input_running_mean,
                             input_running_var,
                             accumulation_count,
                             exp_decay,
                             epsilon,
                             exhaustive_search,
                             deterministic,
                             out_running_mean,
                             out_running_var,
                             saved_mean,
                             saved_var,
                             eq_scale,
                             eq_bias);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_scale_bias_relu_conv_bn,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedScaleBiasReluConvBnKernel,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
}
