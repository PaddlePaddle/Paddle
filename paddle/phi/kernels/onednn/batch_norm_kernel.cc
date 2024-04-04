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

#include "paddle/phi/kernels/batch_norm_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T, typename Context>
void BatchNormKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     const DenseTensor &mean,
                     const DenseTensor &variance,
                     const paddle::optional<DenseTensor> &scale,
                     const paddle::optional<DenseTensor> &bias,
                     bool is_test,
                     float momentum,
                     float epsilon,
                     const std::string &data_layout,
                     bool use_global_stats,
                     bool trainable_statistics,
                     DenseTensor *y,
                     DenseTensor *mean_out,
                     DenseTensor *variance_out,
                     DenseTensor *saved_mean,
                     DenseTensor *saved_variance,
                     DenseTensor *reserve_space) {
  const bool test_mode = is_test && (!trainable_statistics);
  const bool global_stats = test_mode || use_global_stats;
  const bool fuse_with_relu =
      dev_ctx.HasDnnAttr("fuse_with_relu")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("fuse_with_relu"))
          : false;
  const bool use_scale = scale ? true : false;
  const bool use_bias = bias ? true : false;

  funcs::BatchNormOneDNNHandler<T> handler(dev_ctx.GetEngine(),
                                           dev_ctx.GetPlace(),
                                           &x,
                                           epsilon,
                                           use_scale,
                                           use_bias,
                                           fuse_with_relu,
                                           global_stats,
                                           test_mode);

  auto src_memory = handler.AcquireSrcMemory(&x);
  auto dst_memory = handler.AcquireDstMemory(y);
  auto batch_norm_p = handler.AcquireForwardPrimitive();

  std::shared_ptr<dnnl::memory> mean_memory;
  std::shared_ptr<dnnl::memory> variance_memory;

  // mean and variance can be taken either from input or output Tensor
  if (global_stats) {
    mean_memory = handler.AcquireMeanMemory(&mean);
    variance_memory = handler.AcquireVarianceMemory(&variance);
  } else {
    mean_memory = handler.AcquireMeanMemory(saved_mean);
    variance_memory = handler.AcquireVarianceMemory(saved_variance);
  }

  y->set_mem_desc(dst_memory->get_desc());

  std::shared_ptr<dnnl::memory> scale_memory(nullptr);
  std::shared_ptr<dnnl::memory> shift_memory(nullptr);
  auto Scale = scale.get_ptr();
  auto Bias = bias.get_ptr();
  if (scale) scale_memory = handler.AcquireScaleMemory(Scale);
  if (bias) shift_memory = handler.AcquireShiftMemory(Bias);

  auto &astream = OneDNNContext::tls().get_stream();
  batch_norm_p->execute(astream,
                        {{DNNL_ARG_SRC, *src_memory},
                         {DNNL_ARG_SCALE, *scale_memory},
                         {DNNL_ARG_SHIFT, *shift_memory},
                         {DNNL_ARG_MEAN, *mean_memory},
                         {DNNL_ARG_VARIANCE, *variance_memory},
                         {DNNL_ARG_DST, *dst_memory}});
  astream.wait();

  if (!global_stats) {
    const unsigned int C = common::vectorize(mean.dims())[0];

    // onednn only compute stats for current batch
    // so we need compute momentum stats via Eigen lib
    EigenVectorArrayMap<T> batch_mean_e(dev_ctx.template Alloc<T>(saved_mean),
                                        C);
    EigenVectorArrayMap<T> batch_variance_e(
        dev_ctx.template Alloc<T>(saved_variance), C);

    EigenVectorArrayMap<T> running_mean_e(dev_ctx.template Alloc<T>(mean_out),
                                          C);
    EigenVectorArrayMap<T> running_variance_e(
        dev_ctx.template Alloc<T>(variance_out), C);

    running_mean_e = running_mean_e * momentum + batch_mean_e * (1. - momentum);
    running_variance_e =
        running_variance_e * momentum + batch_variance_e * (1. - momentum);
  }
}

template <typename T, typename Context>
void BatchNormInferKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          const DenseTensor &mean,
                          const DenseTensor &variance,
                          const DenseTensor &scale,
                          const DenseTensor &bias,
                          float momentum,
                          float epsilon,
                          const std::string &data_layout,
                          DenseTensor *y,
                          DenseTensor *mean_out,
                          DenseTensor *variance_out) {
  BatchNormKernel<T, Context>(dev_ctx,
                              x,
                              mean,
                              variance,
                              scale,
                              bias,
                              /*is_test=*/true,
                              momentum,
                              epsilon,
                              data_layout,
                              /*use_global_stats=*/false,
                              /*trainable_statistics=*/false,
                              y,
                              mean_out,
                              variance_out,
                              /*saved_mean*/ nullptr,
                              /*saved_variance*/ nullptr,
                              /*reserve_space=*/nullptr);
}

}  // namespace phi

PD_REGISTER_KERNEL(batch_norm, OneDNN, ONEDNN, phi::BatchNormKernel, float) {}
PD_REGISTER_KERNEL(
    batch_norm_infer, OneDNN, ONEDNN, phi::BatchNormInferKernel, float) {}
