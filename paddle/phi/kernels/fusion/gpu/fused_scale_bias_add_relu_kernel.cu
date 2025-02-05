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

template <typename T, typename Context>
void FusedScaleBiasAddReluKernel(const Context& dev_ctx,
                                 const DenseTensor& x1,
                                 const DenseTensor& scale1,
                                 const DenseTensor& bias1,
                                 const DenseTensor& x2,
                                 const paddle::optional<DenseTensor>& scale2,
                                 const paddle::optional<DenseTensor>& bias2,
                                 bool fuse_dual,
                                 bool exhaustive_search,
                                 DenseTensor* out) {
  PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                    80,
                    common::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        dev_ctx.GetComputeCapability()));

  DenseTensor* y = out;
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kScaleBiasAddRelu);

  // exhaustive search
  exhaustive_search = exhaustive_search || FLAGS_cudnn_exhaustive_search;
  bool deterministic = FLAGS_cudnn_deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_search && deterministic,
                    false,
                    common::errors::InvalidArgument(
                        "Can't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));

  // alloc output variables
  dev_ctx.template Alloc<T>(y);

  // get handles
  auto handle = dev_ctx.cudnn_handle();
  auto workspace_handle = dev_ctx.cudnn_workspace_handle();
  // create tensor descriptors
  cudnnTensorFormat_t layout_format = CUDNN_TENSOR_NHWC;
  auto tensor_format = phi::backends::gpu::ToCudnnDataType(x1.dtype());
  auto tensor_format_math = CUDNN_DATA_FLOAT;
  auto compute_dtype = CUDNN_DATA_FLOAT;

  auto dim_x = phi::backends::gpu::TransformDimOrder(
      common::vectorize<int64_t>(x1.dims()));
  std::vector<int64_t> dim_c(dim_x.size(), 1);
  dim_c[1] = dim_x[1];  //  [1, C, 1, 1]

  std::vector<void*> data_ptrs;
  std::vector<int64_t> uids;
  int64_t uid = 100;

  // inputs
  auto x1_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(x1.data<T>()));
  uids.push_back(uid);

  auto x2_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(x2.data<T>()));
  uids.push_back(uid);

  auto scale1_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(scale1.data<T>()));
  uids.push_back(uid);

  auto bias1_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(const_cast<T*>(bias1.data<T>()));
  uids.push_back(uid);

  // dispensable inputs
  auto scale2_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<T*>(scale2->data<T>()));
    uids.push_back(uid);
  }

  auto bias2_desc = helper::GetGeneralTensorDescriptor(
      dim_c, layout_format, ++uid, 16, tensor_format);
  if (fuse_dual) {
    data_ptrs.push_back(const_cast<T*>(bias2->data<T>()));
    uids.push_back(uid);
  }

  // outputs
  auto y_desc = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format);
  data_ptrs.push_back(y->data<T>());
  uids.push_back(uid);

  // virtual outputs
  auto after_scale1 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  auto after_bias1 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  auto after_scale2 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  auto after_bias2 = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  auto after_add = helper::GetGeneralTensorDescriptor(
      dim_x, layout_format, ++uid, 16, tensor_format_math, true);

  // build ops
  auto scale1_op = helper::MakePointwiseOp(
      CUDNN_POINTWISE_MUL, compute_dtype, x1_desc, scale1_desc, after_scale1);

  auto bias1_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                          compute_dtype,
                                          after_scale1,
                                          bias1_desc,
                                          after_bias1);

  auto scale2_op = helper::MakePointwiseOp(
      CUDNN_POINTWISE_MUL, compute_dtype, x2_desc, scale2_desc, after_scale2);

  auto bias2_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                          compute_dtype,
                                          after_scale2,
                                          bias2_desc,
                                          after_bias2);

  cudnn_frontend::Tensor* tensor_to_add = fuse_dual ? &after_bias2 : &x2_desc;

  auto add_op = helper::MakePointwiseOp(CUDNN_POINTWISE_ADD,
                                        compute_dtype,
                                        after_bias1,
                                        *tensor_to_add,
                                        after_add);

  auto relu_desc = cudnn_frontend::PointWiseDescBuilder()
                       .setMode(CUDNN_POINTWISE_RELU_FWD)
                       .setComputeType(compute_dtype)
                       .build();

  auto relu_op = cudnn_frontend::OperationBuilder(
                     CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                     .setxDesc(after_add)
                     .setyDesc(y_desc)
                     .setpwDesc(relu_desc)
                     .build();

  // build op graph
  std::vector<cudnn_frontend::Operation const*> ops;
  if (fuse_dual) {
    ops = std::vector<cudnn_frontend::Operation const*>(
        {&scale1_op, &bias1_op, &scale2_op, &bias2_op, &add_op, &relu_op});
  } else {
    ops = std::vector<cudnn_frontend::Operation const*>(
        {&scale1_op, &bias1_op, &add_op, &relu_op});
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

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_scale_bias_add_relu,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedScaleBiasAddReluKernel,
                   phi::dtype::float16) {}
