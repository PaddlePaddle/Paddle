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

#include "paddle/phi/kernels/pool_grad_kernel.h"

#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/autotune/cache.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/gpudnn/pool_gpudnn.h"
#include "paddle/phi/kernels/pool_kernel.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/kernels/impl/pool_grad_kernel_impl.h"  //  PoolGradRawGPUDNNKernel will call PoolGradRawKernel for pooling type "max" in ROCm
#endif

#ifdef PADDLE_WITH_CUDNN_FRONTEND
#include "paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h"
#endif

PHI_DECLARE_bool(cudnn_deterministic);
PHI_DECLARE_bool(cudnn_exhaustive_search);

namespace phi {

template <typename T, typename Context>
void PoolGradRawGPUDNNKernel(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             const std::vector<int>& kernel_size,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             bool exclusive,
                             const std::string& data_format,
                             const std::string& pooling_type,
                             bool global_pooling,
                             bool adaptive,
                             const std::string& padding_algorithm,
                             DenseTensor* dx) {
  PADDLE_ENFORCE_EQ(
      ctx.GetPlace().GetType() == phi::AllocationType::GPU,
      true,
      errors::InvalidArgument("Pool operator CUDA kernel must use CUDAPlace "
                              "rather than CPUPlace."));

  const DenseTensor* input = &x;
  const DenseTensor* output = &out;
  const DenseTensor* output_grad = &dout;
  DenseTensor* input_grad = dx;
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

#ifdef PADDLE_WITH_HIP
  if (pooling_type == "max") {
    PoolGradRawKernel<T, GPUContext>(ctx,
                                     x,
                                     out,
                                     dout,
                                     kernel_size,
                                     strides,
                                     paddings_,
                                     exclusive,
                                     data_format,
                                     pooling_type,
                                     global_pooling,
                                     adaptive,
                                     padding_algorithm,
                                     dx);
    return;
  }
#endif

  // update paddings
  auto in_x_dims = input->dims();
  DDim data_dims;
  if (channel_last) {
    data_dims = slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
  } else {
    data_dims = slice_ddim(in_x_dims, 2, in_x_dims.size());
  }
  funcs::UpdatePadding(&paddings_,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size_);
  if (data_dims.size() * 2 == static_cast<int>(paddings_.size())) {
    for (int i = 0; i < data_dims.size(); ++i) {
      paddings_.erase(paddings_.begin() + i + 1);
    }
  }

  if (global_pooling) {
    funcs::UpdateKernelSize(&kernel_size_, data_dims);
  }

  // ------- tensor grad --------------
  DenseTensor transformed_input(input->type());
  DenseTensor transformed_output(output->type());
  DenseTensor transformed_output_grad(output_grad->type());

  ctx.template Alloc<T>(input_grad);
  DenseTensor transformed_input_grad(input_grad->type());
  GPUDNNDataLayout layout;
  const std::string str_NCHW = "NCHW", str_NHWC = "NHWC";
  const std::string str_NCDHW = "NCDHW", str_NDHWC = "NDHWC";
  if (data_format == str_NDHWC) {
    layout = GPUDNNDataLayout::kNCDHW;
    std::vector<int> axis{0, 4, 1, 2, 3};

    // input
    transformed_input.Resize(input->dims());
    auto in_dims_vec = common::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[4];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    in_dims_vec[4] = input->dims()[3];
    transformed_input.Resize(common::make_ddim(in_dims_vec));
    ctx.Alloc(&transformed_input, input->type());

    funcs::Transpose<Context, T, 5> trans5;
    trans5(ctx, *input, &transformed_input, axis);

    // output
    transformed_output.Resize(output->dims());
    auto out_dims_vec = common::vectorize(output->dims());
    out_dims_vec[1] = output->dims()[4];
    out_dims_vec[2] = output->dims()[1];
    out_dims_vec[3] = output->dims()[2];
    out_dims_vec[4] = output->dims()[3];
    transformed_output.Resize(common::make_ddim(out_dims_vec));

    ctx.Alloc(&transformed_output, output->type());

    funcs::Transpose<Context, T, 5> trans5_v2;
    trans5_v2(ctx, *output, &transformed_output, axis);

    // output grad
    transformed_output_grad.Resize(common::make_ddim(out_dims_vec));
    ctx.Alloc(&transformed_output_grad, output_grad->type());

    funcs::Transpose<Context, T, 5> trans5_v3;
    trans5_v3(ctx, *output_grad, &transformed_output_grad, axis);

    // input grad
    transformed_input_grad.Resize(common::make_ddim(in_dims_vec));

#ifdef PADDLE_WITH_HIP
    // MIOPEN not support NHWC data layout
  } else if (data_format == str_NHWC) {
    layout = GPUDNNDataLayout::kNCHW;

    std::vector<int> axis{0, 3, 1, 2};

    // input
    transformed_input.Resize(input->dims());
    auto in_dims_vec = common::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[3];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    transformed_input.Resize(common::make_ddim(in_dims_vec));
    ctx.Alloc(&transformed_input, input->type());

    funcs::Transpose<Context, T, 4> trans4;
    trans4(ctx, *input, &transformed_input, axis);

    // output
    transformed_output.Resize(output->dims());
    auto out_dims_vec = common::vectorize(output->dims());
    out_dims_vec[1] = output->dims()[3];
    out_dims_vec[2] = output->dims()[1];
    out_dims_vec[3] = output->dims()[2];
    transformed_output.Resize(common::make_ddim(out_dims_vec));
    ctx.Alloc(&transformed_output, output->type());

    funcs::Transpose<Context, T, 4> trans4_v2;
    trans4_v2(ctx, *output, &transformed_output, axis);

    // output grad
    transformed_output_grad.Resize(common::make_ddim(out_dims_vec));
    ctx.Alloc(&transformed_output_grad, output_grad->type());

    funcs::Transpose<Context, T, 4> trans4_v3;
    trans4_v3(ctx, *output_grad, &transformed_output_grad, axis);

    // input grad
    transformed_input_grad.Resize(common::make_ddim(in_dims_vec));
#endif
  } else {
    layout = GetLayoutFromStr(data_format);
    transformed_input = *input;
    transformed_output = *output;
    transformed_output_grad = *output_grad;
    transformed_input_grad = *input_grad;
  }

  const T* input_data = transformed_input.data<T>();
  const T* output_data = transformed_output.data<T>();
  const T* output_grad_data = transformed_output_grad.data<T>();

  // ------------------- cudnn descriptors ---------------------
  ScopedTensorDescriptor input_desc;
  ScopedTensorDescriptor output_desc;
  ScopedPoolingDescriptor pool_desc;

#ifdef PADDLE_WITH_HIP
  miopenTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
      layout, common::vectorize<int>(transformed_input.dims()));
  miopenTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
      layout, common::vectorize<int>(transformed_output.dims()));
#else
  cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
      layout, common::vectorize<int>(transformed_input.dims()));
  cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
      layout, common::vectorize<int>(transformed_output.dims()));
#endif
  PoolingMode pooling_mode;
  if (pooling_type == "max") {
    if (FLAGS_cudnn_deterministic) {
      pooling_mode = PoolingMode::kMaximumDeterministic;
    } else {
      pooling_mode = PoolingMode::kMaximum;
    }
  } else {
    pooling_mode = exclusive ? PoolingMode::kAverageExclusive
                             : PoolingMode::kAverageInclusive;
  }

#ifdef PADDLE_WITH_HIP
  miopenPoolingDescriptor_t cudnn_pool_desc =
      pool_desc.descriptor(pooling_mode, kernel_size_, paddings_, strides);
#else
  cudnnPoolingDescriptor_t cudnn_pool_desc =
      pool_desc.descriptor(pooling_mode, kernel_size_, paddings_, strides);
#endif

  // ------------------- cudnn pool algorithm ---------------------
  auto handle = ctx.cudnn_handle();
  ScalingParamType<T> alpha = 1.0f, beta = 0.0f;
  if (input_grad) {
    T* input_grad_data = ctx.template Alloc<T>(&transformed_input_grad);
// Because beta is zero, it is unnecessary to reset input_grad.
#ifdef PADDLE_WITH_HIP
    char* pool_workspace;
    size_t pool_worksize = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenPoolingGetWorkSpaceSizeV2(
        cudnn_pool_desc, cudnn_output_desc, &pool_worksize));
    PADDLE_ENFORCE_GPU_SUCCESS(hipMalloc(&pool_workspace, pool_worksize));
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::miopenPoolingBackward(handle,
                                                              cudnn_pool_desc,
                                                              &alpha,
                                                              cudnn_output_desc,
                                                              output_data,
                                                              cudnn_output_desc,
                                                              output_grad_data,
                                                              cudnn_input_desc,
                                                              input_data,
                                                              &beta,
                                                              cudnn_input_desc,
                                                              input_grad_data,
                                                              pool_workspace));
    PADDLE_ENFORCE_GPU_SUCCESS(hipFree(pool_workspace));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cudnnPoolingBackward(handle,
                                                             cudnn_pool_desc,
                                                             &alpha,
                                                             cudnn_output_desc,
                                                             output_data,
                                                             cudnn_output_desc,
                                                             output_grad_data,
                                                             cudnn_input_desc,
                                                             input_data,
                                                             &beta,
                                                             cudnn_input_desc,
                                                             input_grad_data));
#endif

    if (data_format == str_NDHWC) {
      std::vector<int> axis{0, 2, 3, 4, 1};
      funcs::Transpose<Context, T, 5> trans5_v4;
      trans5_v4(ctx, transformed_input_grad, input_grad, axis);
    }
#ifdef PADDLE_WITH_HIP
    // MIOPEN not support NHWC data layout
    if (data_format == str_NHWC) {
      std::vector<int> axis{0, 2, 3, 1};
      funcs::Transpose<Context, T, 4> trans4_v4;
      trans4_v4(ctx, transformed_input_grad, input_grad, axis);
    }
#endif
  }
}

template <typename T, typename Context>
void Pool2dGradGPUDNNKernel(const Context& ctx,
                            const DenseTensor& x,
                            const DenseTensor& out,
                            const DenseTensor& dout,
                            const IntArray& kernel_size,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            bool ceil_mode,
                            bool exclusive,
                            const std::string& data_format,
                            const std::string& pooling_type,
                            bool global_pooling,
                            bool adaptive,
                            const std::string& padding_algorithm,
                            DenseTensor* dx) {
  std::vector<int> kernel_size_val(kernel_size.GetData().begin(),
                                   kernel_size.GetData().end());
  PoolGradRawGPUDNNKernel<T, Context>(ctx,
                                      x,
                                      out,
                                      dout,
                                      kernel_size_val,
                                      strides,
                                      paddings,
                                      exclusive,
                                      data_format,
                                      pooling_type,
                                      global_pooling,
                                      adaptive,
                                      padding_algorithm,
                                      dx);
}

template <typename T, typename Context>
void Pool2dDoubleGradGPUDNNKernel(const Context& ctx,
                                  const DenseTensor& x,
                                  const IntArray& kernel_size,
                                  const std::vector<int>& strides,
                                  const std::vector<int>& paddings,
                                  bool ceil_mode,
                                  bool exclusive,
                                  const std::string& data_format,
                                  const std::string& pooling_type,
                                  bool global_pooling,
                                  bool adaptive,
                                  const std::string& padding_algorithm,
                                  DenseTensor* out) {
  if (pooling_type == "max") {
    PADDLE_THROW(
        errors::InvalidArgument("Pool op grad grad only supports avgpool."));
  } else {
    Pool2dGPUDNNKernel<T, Context>(ctx,
                                   x,
                                   kernel_size,
                                   strides,
                                   paddings,
                                   ceil_mode,
                                   exclusive,
                                   data_format,
                                   pooling_type,
                                   global_pooling,
                                   adaptive,
                                   padding_algorithm,
                                   out);
  }
}

template <typename T, typename Context>
void Pool3dGradGPUDNNKernel(const Context& ctx,
                            const DenseTensor& x,
                            const DenseTensor& out,
                            const DenseTensor& dout,
                            const std::vector<int>& kernel_size,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            bool ceil_mode,
                            bool exclusive,
                            const std::string& data_format,
                            const std::string& pooling_type,
                            bool global_pooling,
                            bool adaptive,
                            const std::string& padding_algorithm,
                            DenseTensor* dx) {
  PoolGradRawGPUDNNKernel<T, Context>(ctx,
                                      x,
                                      out,
                                      dout,
                                      kernel_size,
                                      strides,
                                      paddings,
                                      exclusive,
                                      data_format,
                                      pooling_type,
                                      global_pooling,
                                      adaptive,
                                      padding_algorithm,
                                      dx);
}

#ifdef PADDLE_WITH_CUDNN_FRONTEND

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
                    phi::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        ctx.GetComputeCapability()));
  // Additional options
  bool exhaustive_search = FLAGS_cudnn_exhaustive_search;
  bool deterministic = FLAGS_cudnn_deterministic;
  PADDLE_ENFORCE_EQ(exhaustive_search && deterministic,
                    false,
                    phi::errors::InvalidArgument(
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
      phi::errors::InvalidArgument(
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
    PADDLE_ENFORCE_GT(
        _uid.count(name),
        0,
        phi::errors::InvalidArgument("Name %s is unknown.", name));
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
  auto const mode = cudnn_frontend::cudnnResampleMode_t::CUDNN_RESAMPLE_MAXPOOL;
  auto const padding_mode =
      cudnn_frontend::cudnnPaddingMode_t::CUDNN_NEG_INF_PAD;
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
#endif

}  // namespace phi

using phi::dtype::float16;

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
PD_REGISTER_KERNEL(pool2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Pool2dGradGPUDNNKernel,
                   float,
                   float16) {}
PD_REGISTER_KERNEL(pool2d_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Pool2dDoubleGradGPUDNNKernel,
                   float,
                   float16) {}
PD_REGISTER_KERNEL(pool3d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Pool3dGradGPUDNNKernel,
                   float,
                   float16) {}
#else
PD_REGISTER_KERNEL(pool2d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Pool2dGradGPUDNNKernel,
                   float,
                   double,
                   float16) {}
PD_REGISTER_KERNEL(pool2d_double_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Pool2dDoubleGradGPUDNNKernel,
                   float,
                   double,
                   float16) {}
PD_REGISTER_KERNEL(pool3d_grad,
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::Pool3dGradGPUDNNKernel,
                   float,
                   double,
                   float16) {}
#if defined(PADDLE_WITH_CUDNN_FRONTEND) && (CUDNN_VERSION >= 8600)
PD_REGISTER_KERNEL(max_pool2d_v2_grad,  // cuda_only
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::MaxPool2dV2GradCUDNNKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(2).SetDataType(phi::CppTypeToDataType<int>::Type());
}
#endif
#endif
