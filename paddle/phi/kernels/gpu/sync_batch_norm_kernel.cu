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

#include "paddle/phi/kernels/sync_batch_norm_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/sync_batch_norm_utils.h"

namespace phi {

template <typename T, typename Context>
void SyncBatchNormKernel(const Context &ctx,
                         const DenseTensor &x,
                         const DenseTensor &scale,
                         const DenseTensor &bias,
                         const DenseTensor &mean,
                         const DenseTensor &variance,
                         float momentum,
                         float epsilon_f,
                         const std::string &data_layout_str,
                         bool is_test,
                         bool use_global_stats,
                         bool trainable_statistics,
                         bool fuse_with_relu,
                         DenseTensor *y,
                         DenseTensor *mean_out,
                         DenseTensor *variance_out,
                         DenseTensor *saved_mean,
                         DenseTensor *saved_variance,
                         DenseTensor *reserve_space) {
  PADDLE_ENFORCE_EQ(use_global_stats,
                    false,
                    phi::errors::InvalidArgument(
                        "sync_batch_norm doesn't support "
                        "to set use_global_stats True. Please use batch_norm "
                        "in this case."));

  double epsilon = epsilon_f;
  const bool trainable_stats = trainable_statistics;
  const DataLayout layout =
      paddle::framework::StringToDataLayout(data_layout_str);
  bool test_mode = is_test && (!trainable_statistics);
  const auto &x_dims = x.dims();
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

  const T *x_d = x.template data<T>();
  const auto *s_d = scale.template data<BatchNormParamType<T>>();
  const auto *b_d = bias.template data<BatchNormParamType<T>>();

  T *y_d = ctx.template Alloc<T>(y);

  const BatchNormParamType<T> *mean_data = nullptr;
  const BatchNormParamType<T> *var_data = nullptr;

  auto stream = ctx.stream();
  const int block = 512;
  int max_threads = ctx.GetMaxPhysicalThreadCount();

  paddle::memory::AllocationPtr alloc_ptr{nullptr};

  if (test_mode) {
    mean_data = mean.template data<BatchNormParamType<T>>();
    var_data = variance.template data<BatchNormParamType<T>>();
  } else {
    // x, x^2, 1, here 1 is used to calc device num
    // device num also can be got from platform::DeviceContextPool
    const int bytes = (C * 2 + 1) * sizeof(BatchNormParamType<T>);
    alloc_ptr = paddle::memory::Alloc(
        ctx.GetPlace(),
        bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));

    auto *stats = reinterpret_cast<BatchNormParamType<T> *>(alloc_ptr->ptr());
    const int threads = 256;
    int grid = std::min(C, (max_threads + threads - 1) / threads);
    if (layout == paddle::framework::DataLayout::kNCHW) {
      KeLocalStats<T, threads, paddle::framework::DataLayout::kNCHW>
          <<<grid, threads, 0, stream>>>(x_d, N, H * W * D, C, stats);
    } else {
      KeLocalStats<T, threads, paddle::framework::DataLayout::kNHWC>
          <<<grid, threads, 0, stream>>>(x_d, N, H * W * D, C, stats);
    }

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    int global_gid = 0;
    ncclComm_t comm = nullptr;

    if (paddle::distributed::ProcessGroupMapFromGid::getInstance()->has(
            global_gid)) {
      auto *nccl_pg = static_cast<paddle::distributed::ProcessGroupNCCL *>(
          paddle::distributed::ProcessGroupMapFromGid::getInstance()->get(
              global_gid));
      comm = nccl_pg->NCCLComm(x.place());
    } else {
      comm = ctx.nccl_comm();
    }

    if (comm) {
      int dtype = paddle::platform::ToNCCLDataType(
          paddle::framework::TransToProtoVarType(mean_out->dtype()));
      // In-place operation
      PADDLE_ENFORCE_GPU_SUCCESS(paddle::platform::dynload::ncclAllReduce(
          stats,
          stats,
          2 * C + 1,
          static_cast<ncclDataType_t>(dtype),
          ncclSum,
          comm,
          stream));
      VLOG(3) << "Sync result using all reduce";
    }
#endif

    auto *est_mean_data = ctx.template Alloc<BatchNormParamType<T>>(mean_out);
    auto *est_var_data =
        ctx.template Alloc<BatchNormParamType<T>>(variance_out);

    auto *sv_mean_data = ctx.template Alloc<BatchNormParamType<T>>(saved_mean);
    auto *sv_inv_var_data =
        ctx.template Alloc<BatchNormParamType<T>>(saved_variance);

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
  if (layout == paddle::framework::DataLayout::kNCHW) {
    KeNormAffine<T, paddle::framework::DataLayout::kNCHW>
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
    KeNormAffine<T, paddle::framework::DataLayout::kNHWC>
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
