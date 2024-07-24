// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/sync_batch_norm_utils.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sync_batch_norm_kernel.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/common/flags.h"
COMMON_DECLARE_bool(dynamic_static_unified_comm);
#endif

// sparse header
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void SyncBatchNormKernel(const Context& ctx,
                         const DenseTensor& x,
                         const DenseTensor& mean,
                         const DenseTensor& variance,
                         const DenseTensor& scale,
                         const DenseTensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon_f,
                         const std::string& data_layout_str,
                         bool use_global_stats,
                         bool trainable_statistics,
                         DenseTensor* y,
                         DenseTensor* mean_out,
                         DenseTensor* variance_out,
                         DenseTensor* saved_mean,
                         DenseTensor* saved_variance,
                         DenseTensor* reserve_space) {
  PADDLE_ENFORCE_EQ(use_global_stats,
                    false,
                    phi::errors::InvalidArgument(
                        "sync_batch_norm doesn't support "
                        "to set use_global_stats True. Please use batch_norm "
                        "in this case."));

  double epsilon = epsilon_f;
  const bool trainable_stats = trainable_statistics;
  const DataLayout layout = common::StringToDataLayout(data_layout_str);
  bool test_mode = is_test && (!trainable_statistics);
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The Input dim size should be larger than 1."));
  PADDLE_ENFORCE_LE(x_dims.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "The Input dim size should be less than 6."));
  int N, C, H, W, D;
  funcs::ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);
  int x_numel = x.numel();

  const T* x_d = x.template data<T>();
  const auto* s_d = scale.template data<BatchNormParamType<T>>();
  const auto* b_d = bias.template data<BatchNormParamType<T>>();

  T* y_d = ctx.template Alloc<T>(y);

  const BatchNormParamType<T>* mean_data = nullptr;
  const BatchNormParamType<T>* var_data = nullptr;

  auto stream = ctx.stream();
  const int block = 512;
  int max_threads = ctx.GetMaxPhysicalThreadCount();

  phi::Allocator::AllocationPtr alloc_ptr{nullptr};

  if (test_mode) {
    mean_data = mean.template data<BatchNormParamType<T>>();
    var_data = variance.template data<BatchNormParamType<T>>();
  } else {
    // x, x^2, 1, here 1 is used to calc device num
    // device num also can be got from phi::DeviceContextPool
    const int bytes = (C * 2 + 1) * sizeof(BatchNormParamType<T>);
    phi::DenseTensor stats_tensor;
    stats_tensor.Resize({static_cast<int64_t>(bytes)});
    ctx.template Alloc<BatchNormParamType<T>>(&stats_tensor);
    auto* stats_data = stats_tensor.data<BatchNormParamType<T>>();
    auto* stats = reinterpret_cast<BatchNormParamType<T>*>(stats_data);
    const int threads = 512;
    int grid = std::min(C, (max_threads + threads - 1) / threads);
    if (layout == phi::DataLayout::kNCHW) {
      KeLocalStats<T, threads, phi::DataLayout::kNCHW>
          <<<grid, threads, 0, stream>>>(x_d, N, H * W * D, C, stats);
    } else {
      KeLocalStats<T, threads, phi::DataLayout::kNHWC>
          <<<grid, threads, 0, stream>>>(x_d, N, H * W * D, C, stats);
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    ncclComm_t comm = static_cast<ncclComm_t>(detail::GetCCLComm(x.place(), 0));
    if (comm == nullptr) {
      comm = ctx.nccl_comm();
    }

    if (comm) {
      int dtype = phi::ToNCCLDataType(mean_out->dtype());
      // In-place operation
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::ncclAllReduce(stats,
                                      stats,
                                      2 * C + 1,
                                      static_cast<ncclDataType_t>(dtype),
                                      ncclSum,
                                      comm,
                                      stream));
      VLOG(3) << "Sync result using all reduce";
    } else {
      if (FLAGS_dynamic_static_unified_comm) {
        auto comm_ctx =
            static_cast<distributed::NCCLCommContext*>(ctx.GetCommContext());
        if (comm_ctx) {
          comm_ctx->AllReduce(&stats_tensor, stats_tensor, ncclSum, stream);
          VLOG(3) << "Sync result using all reduce";
        }
      }
    }
#endif

    auto* est_mean_data = ctx.template Alloc<BatchNormParamType<T>>(mean_out);
    auto* est_var_data =
        ctx.template Alloc<BatchNormParamType<T>>(variance_out);

    auto* sv_mean_data = ctx.template Alloc<BatchNormParamType<T>>(saved_mean);
    auto* sv_inv_var_data =
        ctx.template Alloc<BatchNormParamType<T>>(saved_variance);

    int64_t reserve_space_size = 0;
    if (reserve_space == nullptr) {
      reserve_space = new DenseTensor();
    }
    reserve_space->Resize({reserve_space_size});
    ctx.template Alloc<T>(reserve_space);

    // Note, Input('Mean')/Input('Variance') share variable with
    // Output('MeanOut')/Output('VarianceOut')
    KeSyncAndMovingStats<T>
        <<<(C + block - 1) / block, block, 0, stream>>>(stats,
                                                        stats + C,
                                                        stats + 2 * C,
                                                        C,
                                                        momentum,
                                                        epsilon,
                                                        sv_mean_data,
                                                        sv_inv_var_data,
                                                        est_mean_data,
                                                        est_var_data);

    mean_data = sv_mean_data;
    var_data = stats + C;
  }

  int grid2 = (std::min(x_numel, max_threads) + block - 1) / block;
  if (layout == phi::DataLayout::kNCHW) {
    KeNormAffine<T, phi::DataLayout::kNCHW>
        <<<grid2, block, 0, stream>>>(x_d,
                                      s_d,
                                      b_d,
                                      mean_data,
                                      var_data,
                                      epsilon,
                                      C,
                                      H * W * D,
                                      x_numel,
                                      y_d);
  } else {
    KeNormAffine<T, phi::DataLayout::kNHWC>
        <<<grid2, block, 0, stream>>>(x_d,
                                      s_d,
                                      b_d,
                                      mean_data,
                                      var_data,
                                      epsilon,
                                      C,
                                      H * W * D,
                                      x_numel,
                                      y_d);
  }
}

template <typename T, typename Context>
void SyncBatchNormGradKernel(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& scale,
                             const DenseTensor& bias,
                             const DenseTensor& saved_mean,
                             const DenseTensor& saved_variance,
                             const paddle::optional<DenseTensor>& reserve_space,
                             const DenseTensor& y_grad,
                             float momentum,
                             float epsilon_f,
                             const std::string& data_layout_str,
                             bool is_test,
                             bool use_global_stats,
                             bool trainable_statistics,
                             DenseTensor* x_grad,
                             DenseTensor* scale_grad,
                             DenseTensor* bias_grad) {
  SyncBatchNormGradFunctor<T, Context>(ctx,
                                       &x,
                                       nullptr,
                                       scale,
                                       bias,
                                       saved_mean,
                                       saved_variance,
                                       y_grad,
                                       epsilon_f,
                                       data_layout_str,
                                       x_grad,
                                       scale_grad,
                                       bias_grad);
}

}  // namespace phi

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SyncBatchNormCooKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& mean,
                            const DenseTensor& variance,
                            const DenseTensor& scale,
                            const DenseTensor& bias,
                            bool is_test,
                            float momentum,
                            float epsilon,
                            const std::string& data_layout,
                            bool use_global_stats,
                            bool trainable_statistics,
                            SparseCooTensor* y,
                            DenseTensor* mean_out,
                            DenseTensor* variance_out,
                            DenseTensor* saved_mean,
                            DenseTensor* saved_variance,
                            DenseTensor* reserve_space) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, y);
  phi::SyncBatchNormKernel<T, Context>(dev_ctx,
                                       x.values(),
                                       mean,
                                       variance,
                                       scale,
                                       bias,
                                       is_test,
                                       momentum,
                                       epsilon,
                                       data_layout,
                                       use_global_stats,
                                       trainable_statistics,
                                       y->mutable_values(),
                                       mean_out,
                                       variance_out,
                                       saved_mean,
                                       saved_variance,
                                       reserve_space);
  y->SetIndicesDict(x.GetIndicesDict());
  y->SetKmaps(x.GetKmaps());
}

template <typename T, typename Context>
void SyncBatchNormCooGradKernel(
    const Context& dev_ctx,
    const SparseCooTensor& x,
    const DenseTensor& scale,
    const DenseTensor& bias,
    const DenseTensor& saved_mean,
    const DenseTensor& saved_variance,
    const paddle::optional<DenseTensor>& reserve_space,
    const SparseCooTensor& y_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    SparseCooTensor* x_grad,
    DenseTensor* scale_grad,
    DenseTensor* bias_grad) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, x_grad);
  *scale_grad = phi::EmptyLike<T, Context>(dev_ctx, scale);
  *bias_grad = phi::EmptyLike<T, Context>(dev_ctx, bias);
  phi::SyncBatchNormGradKernel<T, Context>(dev_ctx,
                                           x.values(),
                                           scale,
                                           bias,
                                           saved_mean,
                                           saved_variance,
                                           reserve_space,
                                           y_grad.values(),
                                           momentum,
                                           epsilon,
                                           data_layout,
                                           is_test,
                                           use_global_stats,
                                           trainable_statistics,
                                           x_grad->mutable_values(),
                                           scale_grad,
                                           bias_grad);
}

}  // namespace sparse
}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(sync_batch_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormKernel,
                   float,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(sync_batch_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
}
#else
PD_REGISTER_KERNEL(sync_batch_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif
#endif

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(sync_batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormGradKernel,
                   float,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(sync_batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(sync_batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
#endif

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(sync_batch_norm_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SyncBatchNormCooKernel,
                   float,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(sync_batch_norm_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SyncBatchNormCooKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(sync_batch_norm_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SyncBatchNormCooGradKernel,
                   float,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(sync_batch_norm_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SyncBatchNormCooGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
