// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/collective/process_group_bkcl.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "xpu/bkcl.h"

namespace phi {

template <typename T, typename Context>
void KeLocalStats_xpu(const Context& dev_ctx,
                      const DataLayout& layout,
                      const int64_t& N,
                      const int64_t& C,
                      const int64_t& H,
                      const int64_t& W,
                      const phi::DenseTensor* x,
                      phi::DenseTensor* mean_var) {
  int64_t x_numel = x->numel();
  std::vector<int64_t> x_shape = phi::vectorize<int64_t>(x->dims());
  std::vector<int64_t> rdims;
  if (layout == phi::DataLayout::kNCHW) {
    rdims = {0, 2, 3};
  } else if (layout == phi::DataLayout::kNHWC) {
    rdims = {0, 1, 2};
  }

  std::vector<int64_t> c_shape;
  if (layout == phi::DataLayout::kNCHW)
    c_shape = {1, C, 1, 1};
  else if (layout == phi::DataLayout::kNHWC)
    c_shape = {1, 1, 1, C};

  int r = 0;
  float* mean_var_data = mean_var->data<float>();
  r = xpu::reduce_mean(
      dev_ctx.x_context(), x->data<T>(), mean_var_data, x_shape, rdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_mean");

  phi::DenseTensor x_square;
  x_square.Resize(x->dims());
  dev_ctx.template Alloc<float>(&x_square);

  // Square
  r = xpu::mul(dev_ctx.x_context(),
               x->data<T>(),
               x->data<T>(),
               x_square.data<float>(),
               x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");

  r = xpu::reduce_mean(dev_ctx.x_context(),
                       x_square.data<float>(),
                       mean_var_data + C,
                       x_shape,
                       rdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_mean");

  r = xpu::constant(dev_ctx.x_context(), mean_var_data + C * 2, 1, 1.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
}

template <typename T, typename Context>
void KeSyncAndMovingStats_xpu(const Context& dev_ctx,
                              float* means,
                              float* vars,
                              float device_counts,
                              const int64_t& C,
                              float momentum,
                              float epsilon,
                              float* sv_mean_data,
                              float* sv_inv_var_data,
                              float* moving_means,
                              float* moving_variances) {
  auto xpu_context = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_context);
  float* temp_1 = RAII_GUARD.alloc_l3_or_gm<float>(C);
  PADDLE_ENFORCE_NOT_NULL(
      temp_1, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));
  float* temp_2 = RAII_GUARD.alloc_l3_or_gm<float>(C);
  PADDLE_ENFORCE_NOT_NULL(
      temp_2, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));

  int r = 0;
  r = xpu::scale(dev_ctx.x_context(),
                 means,
                 sv_mean_data,
                 C,
                 false,
                 1.0f / device_counts,
                 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

  float* mean_square = RAII_GUARD.alloc_l3_or_gm<float>(C);
  PADDLE_ENFORCE_NOT_NULL(
      mean_square, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));

  // Square
  r = xpu::mul(dev_ctx.x_context(), sv_mean_data, sv_mean_data, mean_square, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");

  // vars = vars - mean * mean
  r = xpu::scale(
      dev_ctx.x_context(), vars, vars, C, false, 1.0f / device_counts, 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = xpu::sub(dev_ctx.x_context(), vars, mean_square, vars, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sub");

  // sv_inv_var_data[i] = 1.0 / sqrt(var + epsilon);
  r = xpu::scale(dev_ctx.x_context(), vars, temp_1, C, true, 1.0f, epsilon);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = xpu::rsqrt(dev_ctx.x_context(), temp_1, sv_inv_var_data, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "rsqrt");

  // moving_means[i] = moving_means[i] * momentum + mean * (1. - momentum);
  r = xpu::scale(
      dev_ctx.x_context(), moving_means, temp_1, C, false, momentum, 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = xpu::scale(dev_ctx.x_context(),
                 sv_mean_data,
                 temp_2,
                 C,
                 false,
                 1.0f - momentum,
                 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = xpu::add(dev_ctx.x_context(), temp_1, temp_2, moving_means, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");

  // moving_variances[i] = moving_variances[i] * momentum + vars * (1. -
  // momentum);
  r = xpu::scale(
      dev_ctx.x_context(), moving_variances, temp_1, C, false, momentum, 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = xpu::scale(
      dev_ctx.x_context(), vars, temp_2, C, false, 1.0f - momentum, 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = xpu::add(dev_ctx.x_context(), temp_1, temp_2, moving_variances, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
}

template <typename T, typename Context>
void SyncBatchNormKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& mean,
                         const DenseTensor& variance,
                         const DenseTensor& scale,
                         const DenseTensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon,
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

  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(x_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dimension must equal to 4. But "
                        "received X's shape = [%s], X's dimension = [%d].",
                        x_dims,
                        x_dims.size()));

  auto layout = phi::StringToDataLayout(data_layout_str);
  int64_t N, C, H, W;
  if (layout == phi::DataLayout::kNCHW) {
    N = x_dims[0];
    C = x_dims[1];
    H = x_dims[2];
    W = x_dims[3];
  } else {
    N = x_dims[0];
    H = x_dims[1];
    W = x_dims[2];
    C = x_dims[3];
  }

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<float>(mean_out);
  dev_ctx.template Alloc<float>(variance_out);
  dev_ctx.template Alloc<float>(saved_mean);
  dev_ctx.template Alloc<float>(saved_variance);

  bool test_mode = is_test && (!trainable_statistics);
  const float* mean_data = nullptr;
  const float* var_data = nullptr;
  if (test_mode) {
    mean_data = mean.data<float>();
    var_data = variance.data<float>();
  } else {
    phi::DenseTensor stats;
    stats.Resize({C * 2 + 1});
    float* stats_data = dev_ctx.template Alloc<float>(&stats);

    KeLocalStats_xpu<T, Context>(dev_ctx, layout, N, C, H, W, &x, &stats);

    if (std::getenv("XPU_PADDLE_SYNC_BN_STATIC") != nullptr) {
      auto stream = dev_ctx.stream();
      auto comm = paddle::platform::BKCLCommContext::Instance().Get(
          0, dev_ctx.GetPlace());
      if (comm) {
        auto dtype = paddle::platform::ToBKCLDataType(
            paddle::framework::TransToProtoVarType(stats.dtype()));
        {
          void* sendbuff =
              reinterpret_cast<void*>(const_cast<float*>(stats.data<float>()));
          void* recvbuff = sendbuff;
          PADDLE_ENFORCE_EQ(bkcl_all_reduce(comm->comm(),
                                            sendbuff,
                                            recvbuff,
                                            C * 2 + 1,
                                            dtype,
                                            BKCL_ADD,
                                            stream),
                            BKCL_SUCCESS,
                            paddle::platform::errors::PreconditionNotMet(
                                "bckl all reduce failed"));
        }
      }
    } else {
      auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();
      paddle::distributed::ProcessGroup* pg = map->get(0);
      paddle::distributed::AllreduceOptions opts;
      opts.reduce_op = paddle::distributed::ReduceOp::SUM;

      std::vector<phi::DenseTensor> in_out;
      in_out.push_back(stats);
      pg->AllReduce(in_out, in_out, opts)->Synchronize();
    }

    float device_counts = 0.0;
    phi::memory_utils::Copy(phi::CPUPlace(),
                            static_cast<void*>(&device_counts),
                            dev_ctx.GetPlace(),
                            static_cast<void*>(stats_data + C * 2),
                            sizeof(float));

    KeSyncAndMovingStats_xpu<T, Context>(dev_ctx,
                                         stats_data,
                                         stats_data + C,
                                         device_counts,
                                         C,
                                         momentum,
                                         epsilon,
                                         saved_mean->data<float>(),
                                         saved_variance->data<float>(),
                                         mean_out->data<float>(),
                                         variance_out->data<float>());

    mean_data = saved_mean->data<float>();
    var_data = stats_data + C;
  }

  bool is_nchw = true;
  if (layout == phi::DataLayout::kNHWC) {
    is_nchw = false;
  }
  int r = xpu::batch_norm_infer(dev_ctx.x_context(),
                                x.data<T>(),
                                y->data<T>(),
                                N,
                                C,
                                H,
                                W,
                                epsilon,
                                scale.data<float>(),
                                bias.data<float>(),
                                mean_data,
                                var_data,
                                is_nchw);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "batch_norm_infer");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    sync_batch_norm, XPU, ALL_LAYOUT, phi::SyncBatchNormKernel, float) {}
