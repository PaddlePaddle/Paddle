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
void KeBackwardLocalStats_xpu(const Context &dev_ctx,
                              const DataLayout &layout,
                              const int64_t &C,
                              const phi::DenseTensor *dy,
                              const phi::DenseTensor *x,
                              const phi::DenseTensor *means,
                              phi::DenseTensor *sum_dy_prod) {
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

  float *sum_dy_prod_data = sum_dy_prod->data<float>();
  int r = 0;

  // reduce_sum(dy)
  r = xpu::reduce_sum(
      dev_ctx.x_context(), dy->data<T>(), sum_dy_prod_data, x_shape, rdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

  auto xpu_context = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_context);
  float *temp_1 = RAII_GUARD.alloc_l3_or_gm<float>(x_numel);
  PADDLE_ENFORCE_NOT_NULL(
      temp_1, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));
  float *temp_2 = RAII_GUARD.alloc_l3_or_gm<float>(x_numel);
  PADDLE_ENFORCE_NOT_NULL(
      temp_2, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));

  // reduce_sum(dy * (x - means))
  r = xpu::broadcast<float>(
      dev_ctx.x_context(), means->data<float>(), temp_1, c_shape, x_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
  r = xpu::sub(dev_ctx.x_context(), x->data<T>(), temp_1, temp_2, x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sub");
  r = xpu::mul(dev_ctx.x_context(), dy->data<T>(), temp_2, temp_2, x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  r = xpu::reduce_sum(
      dev_ctx.x_context(), temp_2, sum_dy_prod_data + C, x_shape, rdims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

  // for device_count
  r = xpu::constant(dev_ctx.x_context(), sum_dy_prod_data + C * 2, 1, 1.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
}

template <typename T, typename Context>
void KeBNBackwardData_xpu(const Context &dev_ctx,
                          const DataLayout &layout,
                          const phi::DenseTensor *dy,
                          const phi::DenseTensor *x,
                          const phi::DenseTensor *scale,
                          const phi::DenseTensor *mean,
                          const phi::DenseTensor *var,
                          const float *sum_dy,
                          const float *sum_dy_prod,
                          float device_counts,
                          float epsilon,
                          const int64_t &C,
                          phi::DenseTensor *dx) {
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

  auto xpu_context = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_context);
  float *gvar = RAII_GUARD.alloc_l3_or_gm<float>(x_numel);
  PADDLE_ENFORCE_NOT_NULL(
      gvar, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));
  float *gmean = RAII_GUARD.alloc_l3_or_gm<float>(x_numel);
  PADDLE_ENFORCE_NOT_NULL(
      gmean, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));

  float *temp1 = RAII_GUARD.alloc_l3_or_gm<float>(x_numel);
  PADDLE_ENFORCE_NOT_NULL(
      temp1, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));
  float *temp2 = RAII_GUARD.alloc_l3_or_gm<float>(x_numel);
  PADDLE_ENFORCE_NOT_NULL(
      temp2, phi::errors::External("XPU alloc_l3_or_gm returns nullptr"));

  // gvar = (g_sum_dy_prod[c] / dev_num) * s_d * inv_var * (inv_var * inv_var) *
  // C / x_numel;
  r = xpu::mul(
      dev_ctx.x_context(), scale->data<float>(), var->data<float>(), temp1, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  r = xpu::mul(dev_ctx.x_context(), sum_dy_prod, temp1, temp2, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  r = xpu::mul(dev_ctx.x_context(), var->data<float>(), temp2, temp2, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  r = xpu::mul(dev_ctx.x_context(), var->data<float>(), temp2, temp2, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  r = xpu::scale(
      dev_ctx.x_context(), temp2, temp2, C, false, 1.0f * C / x_numel, 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = xpu::broadcast<float>(dev_ctx.x_context(), temp2, gvar, c_shape, x_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

  // gmean = (g_sum_dy[c] / dev_num) * s_d * inv_var * C / x_numel;
  r = xpu::mul(dev_ctx.x_context(), sum_dy, temp1, temp2, C);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  r = xpu::scale(
      dev_ctx.x_context(), temp2, temp2, C, false, 1.0f * C / x_numel, 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  r = xpu::broadcast<float>(
      dev_ctx.x_context(), temp2, gmean, c_shape, x_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

  // dy * scale * var
  r = xpu::broadcast<float>(
      dev_ctx.x_context(), temp1, temp2, c_shape, x_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
  r = xpu::mul(dev_ctx.x_context(), dy->data<T>(), temp2, temp2, x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");

  // dy * scale * var - gmean
  r = xpu::sub(dev_ctx.x_context(), temp2, gmean, temp2, x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sub");

  // x - mean
  r = xpu::broadcast<float>(
      dev_ctx.x_context(), mean->data<float>(), temp1, c_shape, x_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");
  r = xpu::sub(dev_ctx.x_context(), x->data<T>(), temp1, temp1, x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sub");

  // gvar * (x - mean)
  r = xpu::mul(dev_ctx.x_context(), gvar, temp1, temp1, x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");

  // dx = dy * scale * var - gmean - gvar * (x - mean);
  r = xpu::sub(dev_ctx.x_context(), temp2, temp1, dx->data<T>(), x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sub");
}

template <typename T, typename Context>
void SyncBatchNormGradKernel(const Context &dev_ctx,
                             const DenseTensor &x,
                             const DenseTensor &scale,
                             const DenseTensor &bias,
                             const DenseTensor &saved_mean,
                             const DenseTensor &saved_variance,
                             const paddle::optional<DenseTensor> &reserve_space,
                             const DenseTensor &y_grad,
                             float momentum,
                             float epsilon_f,
                             const std::string &data_layout_str,
                             bool is_test,
                             bool use_global_stats,
                             bool trainable_statistics,
                             DenseTensor *x_grad,
                             DenseTensor *scale_grad,
                             DenseTensor *bias_grad) {
  const auto &x_dims = x.dims();
  auto layout = phi::StringToDataLayout(data_layout_str);
  int64_t C;
  if (layout == phi::DataLayout::kNCHW) {
    C = x_dims[1];
  } else {
    C = x_dims[3];
  }

  phi::DenseTensor stats;
  stats.Resize({C * 2 + 1});
  float *stats_data = dev_ctx.template Alloc<float>(&stats);

  KeBackwardLocalStats_xpu<T, Context>(
      dev_ctx, layout, C, &y_grad, &x, &saved_mean, &stats);

  int r = 0;
  if (scale_grad && bias_grad) {
    // d_bias = reduce_sum(dy)
    dev_ctx.template Alloc<float>(bias_grad);
    r = xpu::copy(dev_ctx.x_context(), stats_data, bias_grad->data<float>(), C);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

    // d_scale = reduce_sum(dy * (xy - saved_mean)) * saved_variance
    dev_ctx.template Alloc<float>(scale_grad);
    r = xpu::mul(dev_ctx.x_context(),
                 stats_data + C,
                 saved_variance.data<float>(),
                 scale_grad->data<float>(),
                 C);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  }

  if (std::getenv("XPU_PADDLE_SYNC_BN_STATIC") != nullptr) {
    auto stream = dev_ctx.stream();
    auto comm = paddle::platform::BKCLCommContext::Instance().Get(
        0, dev_ctx.GetPlace());
    if (comm) {
      auto dtype = paddle::platform::ToBKCLDataType(
          paddle::framework::TransToProtoVarType(stats.dtype()));
      // BkclAllReduce
      {
        void *sendbuff =
            reinterpret_cast<void *>(const_cast<float *>(stats.data<float>()));
        void *recvbuff = sendbuff;
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
    paddle::distributed::ProcessGroup *pg = map->get(0);
    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = paddle::distributed::ReduceOp::SUM;

    std::vector<phi::DenseTensor> in_out;
    in_out.push_back(stats);
    pg->AllReduce(in_out, in_out, opts)->Synchronize();
  }

  float device_counts = 0.0;
  phi::memory_utils::Copy(phi::CPUPlace(),
                          static_cast<void *>(&device_counts),
                          dev_ctx.GetPlace(),
                          static_cast<void *>(stats_data + C * 2),
                          sizeof(float));

  r = xpu::scale(dev_ctx.x_context(),
                 stats_data,
                 stats_data,
                 C * 2,
                 false,
                 1.0f / device_counts,
                 0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

  dev_ctx.template Alloc<float>(x_grad);
  KeBNBackwardData_xpu<T, Context>(dev_ctx,
                                   layout,
                                   &y_grad,
                                   &x,
                                   &scale,
                                   &saved_mean,
                                   &saved_variance,
                                   stats_data,
                                   stats_data + C,
                                   device_counts,
                                   epsilon_f,
                                   C,
                                   x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(sync_batch_norm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormGradKernel,
                   float) {}
