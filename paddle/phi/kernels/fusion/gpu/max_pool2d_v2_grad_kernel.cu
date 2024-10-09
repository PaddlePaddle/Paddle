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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"
#include "paddle/phi/kernels/gpudnn/pool_gpudnn.h"

COMMON_DECLARE_bool(cudnn_exhaustive_search);

namespace phi {

template <typename Context, typename T1, typename T2 = int>
void MaxPoolV2GradCUDNNKernel(const Context& ctx,
                              const DenseTensor& x,
                              const DenseTensor& out,
                              const DenseTensor& saved_idx,
                              const DenseTensor& dout,
                              const std::vector<int>& kernel_size,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::string& data_format,
                              bool global_pooling,
                              bool adaptive,
                              DenseTensor* dx) {
  PADDLE_ENFORCE_GE(ctx.GetComputeCapability(),
                    80,
                    common::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        ctx.GetComputeCapability()));
  // Additional options
  bool exhaustive_search = FLAGS_cudnn_exhaustive_search;
  bool deterministic = FLAGS_cudnn_deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_search && deterministic,
                    false,
                    common::errors::InvalidArgument(
                        "Can't set exhaustive_search True and "
                        "FLAGS_cudnn_deterministic True at same time."));
  // Allocate output tensors
  ctx.template Alloc<T1>(dx);
  // Update paddings
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;
  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
  PADDLE_ENFORCE_EQ(
      channel_last,
      true,
      common::errors::InvalidArgument(
          "NCHW layout is currently not supported for max pooling bwd."));
  const std::string padding_algorithm = "EXPLICIT";

  auto x_dims = x.dims();
  DDim data_dims;
  if (channel_last) {
    data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  } else {
    data_dims = slice_ddim(x_dims, 2, x_dims.size());
  }
  funcs::UpdatePadding(&paddings_,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size_);

  const auto data_dim = data_dims.size();
  std::vector<int64_t> pre_padding(data_dim, 0);
  std::vector<int64_t> post_padding(data_dim, 0);
  for (size_t i = 0; i < data_dim; ++i) {
    pre_padding[i] = static_cast<int64_t>(paddings_[2 * i]);
    post_padding[i] = static_cast<int64_t>(paddings_[2 * i + 1]);
  }

  if (global_pooling) {
    funcs::UpdateKernelSize(&kernel_size_, data_dims);
  }

  using helper = CudnnFrontendConvHelper;
  auto kernel_size_int64 = helper::GetInt64Array(kernel_size_);
  auto strides_int64 = helper::GetInt64Array(strides);

  // Create tensor descriptors
  auto& plan_cache = phi::autotune::AutoTuneCache::Instance().GetConvV8(
      phi::autotune::AlgorithmType::kPoolingBackwardV8);

  T2* saved_idx_data = const_cast<T2*>(saved_idx.data<T2>());
  T1* dout_data = const_cast<T1*>(dout.data<T1>());
  T1* dx_data = dx->data<T1>();

  auto uid = [](std::string name) {
    const std::map<std::string, int64_t> _uid = {
        {"saved_idx", 0}, {"dout", 1}, {"dx", 2}};
    PADDLE_ENFORCE_GT(_uid.count(name),
                      0,
                      common::errors::InvalidArgument(
                          "The tensor name %s is unknown. "
                          "Should be in one of [saved_idx, dout, dx].",
                          name));
    return _uid.at(name);
  };

  cudnnHandle_t handle = const_cast<cudnnHandle_t>(ctx.cudnn_handle());
  auto workspace_handle = ctx.cudnn_workspace_handle();

  auto layout = GetLayoutFromStr(data_format);
  auto layout_format = phi::backends::gpu::GetCudnnTensorFormat(layout);
  auto input_dtype = phi::backends::gpu::CudnnDataType<T1>::type;
  auto saved_idx_dtype = CudnnIndexType<T2>::type;

  // Create plan and execute
  std::vector<void*> data_ptrs({saved_idx_data, dout_data, dx_data});
  std::vector<int64_t> uids({uid("saved_idx"), uid("dout"), uid("dx")});

  // Create feature vector for plan caching
  cudnn_frontend::feature_vector_t feature_vector;
  auto dim_x = phi::vectorize<int64_t>(x.dims());
  phi::autotune::BuildFeatureVector(&feature_vector,
                                    dim_x,
                                    kernel_size_int64,
                                    strides_int64,
                                    pre_padding,
                                    post_padding,
                                    data_format,
                                    input_dtype,
                                    saved_idx_dtype);

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

  auto saved_idx_desc =
      helper::GetTensorDescriptor(&saved_idx, uid("saved_idx"), layout_format);
  auto dout_desc =
      helper::GetTensorDescriptor(&dout, uid("dout"), layout_format);
  auto dx_desc = helper::GetTensorDescriptor(dx, uid("dx"), layout_format);

  // Create maxpooling descriptor
  auto const nan_opt = CUDNN_NOT_PROPAGATE_NAN;
  auto const mode = cudnn_frontend::ResampleMode_t::MAXPOOL;
  auto const padding_mode = cudnn_frontend::PaddingMode_t::NEG_INF_PAD;
  auto pool_desc = cudnn_frontend::ResampleDescBuilder_v8()
                       .setComputeType(CUDNN_DATA_FLOAT)
                       .setNanPropagation(nan_opt)
                       .setResampleMode(mode)
                       .setPaddingMode(padding_mode)
                       .setSpatialDim(data_dim, kernel_size_int64.data())
                       .setSpatialStride(data_dim, strides_int64.data())
                       .setPrePadding(data_dim, pre_padding.data())
                       .setPostPadding(data_dim, post_padding.data())
                       .build();

  // Create maxpooling bwd op
  auto pool_op = cudnn_frontend::OperationBuilder(
                     CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                     .setdxDesc(dx_desc)
                     .setdyDesc(dout_desc)
                     .setidxDesc(saved_idx_desc)
                     .setResampleDesc(pool_desc)
                     .build();

  // Create op graph
  std::array<cudnn_frontend::Operation const*, 1> ops = {&pool_op};
  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(handle)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();

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
void MaxPool2dV2GradCUDNNKernel(const Context& ctx,
                                const DenseTensor& x,
                                const DenseTensor& out,
                                const DenseTensor& saved_idx,
                                const DenseTensor& dout,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::string& data_format,
                                bool global_pooling,
                                bool adaptive,
                                DenseTensor* dx) {
  MaxPoolV2GradCUDNNKernel<Context, T>(ctx,
                                       x,
                                       out,
                                       saved_idx,
                                       dout,
                                       kernel_size,
                                       strides,
                                       paddings,
                                       data_format,
                                       global_pooling,
                                       adaptive,
                                       dx);
}

}  // namespace phi

using phi::dtype::float16;

PD_REGISTER_KERNEL(max_pool2d_v2_grad,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MaxPool2dV2GradCUDNNKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(2).SetDataType(phi::CppTypeToDataType<int>::Type());
}
