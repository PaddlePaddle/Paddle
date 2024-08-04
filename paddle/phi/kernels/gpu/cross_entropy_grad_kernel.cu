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

#include "paddle/phi/kernels/cross_entropy_grad_kernel.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {

template <typename T>
__global__ void SoftLabelCrossEntropyGradientKernel(T* logit_grad,
                                                    const T* loss_grad,
                                                    const T* labels,
                                                    const int n,
                                                    const int d,
                                                    const int remain) {
  int ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int idx_n = ids / d;
    int idx_remain = ids % remain;
    int idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (-labels[ids] / logit_grad[ids]);
  }
}

template <typename T, typename LabelT>
__global__ void HardLabelCrossEntropyGradientKernel(T* logit_grad,
                                                    const LabelT* labels,
                                                    const int n,
                                                    const int d,
                                                    const int remain,
                                                    const int ignore_index) {
  CUDA_KERNEL_LOOP(index, n * remain) {
    int idx_n = index / remain;
    int idx_remain = index % remain;
    int tmp = static_cast<int>(labels[index]);
    int idx = idx_n * d + tmp * remain + idx_remain;
    if (ignore_index != tmp) {
      logit_grad[idx] = -static_cast<T>(1.) / logit_grad[idx];
    }
  }
}

template <typename T, typename LabelT>
__global__ void ScaleCrossEntropyGradient(T* logit_grad,
                                          const T* loss_grad,
                                          const int num,
                                          const int d,
                                          const int remain,
                                          const LabelT* labels,
                                          const int ignore_index) {
  CUDA_KERNEL_LOOP(index, num) {
    int idx_n = index / d;
    int idx_remain = index % remain;
    int idx_lbl = idx_n * remain + idx_remain;
    int k = (index % d) / remain;
    auto lbl = static_cast<int64_t>(labels[idx_lbl]);
    if (lbl == ignore_index || lbl != k) {
      logit_grad[index] = static_cast<T>(0.);
    } else {
      logit_grad[index] *= loss_grad[idx_lbl];
    }
  }
}

template <typename T>
__global__ void SoftCrossEntropyGradientKernel(T* logit_grad,
                                               const T* loss_grad,
                                               const T* labels,
                                               const int64_t n,
                                               const int64_t d,
                                               const int64_t remain) {
  int64_t ids = blockIdx.x * blockDim.x + threadIdx.x;
  if (ids < n * d) {
    int64_t idx_n = ids / d;
    int64_t idx_remain = ids % remain;
    int64_t idx_loss = idx_n * remain + idx_remain;
    logit_grad[ids] = loss_grad[idx_loss] * (logit_grad[ids] - labels[ids]);
  }
}

/*
  Wrapper of softmax with cross entropy grad hard label.
*/
template <typename T, typename LabelT>
__global__ void SoftmaxWithCrossEntropyGradHardLabel(T* logits_grad,
                                                     const T* loss_grad,
                                                     const T* softmax,
                                                     const LabelT* labels,
                                                     const int64_t n,
                                                     const int64_t dim,
                                                     const int64_t d,
                                                     const int ignore_index) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t idx_n = idx / (d * dim);
  int64_t idx_dim = (idx / d) % dim;
  int64_t idx_d = idx % d;
  int64_t ids = idx_n * d + idx_d;

  if (idx < n * dim * d) {
    auto lbl = static_cast<int64_t>(labels[ids]);
    if (lbl == ignore_index) {
      logits_grad[idx] = static_cast<T>(0.0);
    } else if (lbl == idx_dim) {
      logits_grad[idx] = (softmax[idx] - static_cast<T>(1.0)) * loss_grad[ids];
    } else {
      logits_grad[idx] = softmax[idx] * loss_grad[ids];
    }
  }
}

template <typename T, typename LabelT>
void CrossEntropyWithSoftmaxGradGPUKernel(const GPUContext& dev_ctx,
                                          const DenseTensor& label,
                                          const DenseTensor& softmax,
                                          const DenseTensor& loss_grad,
                                          bool soft_label,
                                          bool use_softmax,
                                          bool numeric_stable_mode,
                                          int ignore_index,
                                          int axis,
                                          DenseTensor* logits_grad) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType(),
      phi::AllocationType::GPU,
      common::errors::Unavailable("softmax_with_cross_entropy operator's "
                                  "CUDA kernel only runs on GPU device."));
  const T* loss_grad_data = loss_grad.data<T>();
  DenseTensor* logit_grad = logits_grad;

  T* logit_grad_data = nullptr;
  bool copy_flag = (logit_grad != &softmax && (!use_softmax || soft_label));
  if (copy_flag) {
    phi::Copy(dev_ctx, softmax, dev_ctx.GetPlace(), false, logit_grad);
    logit_grad_data = logit_grad->data<T>();
  } else {
    logit_grad_data = dev_ctx.template Alloc<T>(logit_grad);
  }

  const int rank = logit_grad->dims().size();
  const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = logit_grad->dims()[axis_v];

  const int64_t n = phi::funcs::SizeToAxis(axis_v, logit_grad->dims());
  const int64_t d = phi::funcs::SizeFromAxis(axis_v, logit_grad->dims());
  const int64_t remain = d / axis_dim;

  int block = 512;
  auto stream = dev_ctx.stream();

  // do not with softmax op, and input is softmax
  if (!use_softmax) {
    if (soft_label) {
      int grid = (n * d + block - 1) / block;
      const T* label_data = label.data<T>();
      SoftLabelCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
          logit_grad_data, loss_grad_data, label_data, n, d, remain);
    } else {
      DenseTensor logits_grad_2d(*logit_grad);
      logits_grad_2d.Resize({n, d});
      int grid = (n * remain + block - 1) / block;
      const auto* label_data = label.data<LabelT>();
      HardLabelCrossEntropyGradientKernel<T, LabelT>
          <<<grid, block, 0, stream>>>(
              logit_grad_data, label_data, n, d, remain, ignore_index);
      int num = n * d;
      grid = (num + block - 1) / block;
      ScaleCrossEntropyGradient<T, LabelT>
          <<<grid, block, 0, stream>>>(logit_grad_data,
                                       loss_grad_data,
                                       num,
                                       d,
                                       remain,
                                       label_data,
                                       ignore_index);
    }

    return;
  }

  // with softmax, continue

  if (soft_label) {
    int64_t grid = (n * d + block - 1) / block;
    const T* label_data = label.data<T>();
    SoftCrossEntropyGradientKernel<T><<<grid, block, 0, stream>>>(
        logit_grad_data, loss_grad_data, label_data, n, d, remain);
  } else {
    const T* softmax_data = softmax.data<T>();
    const auto* label_data = label.data<LabelT>();
    int grid = (n * d + block - 1) / block;
    SoftmaxWithCrossEntropyGradHardLabel<T>
        <<<grid, block, 0, stream>>>(logit_grad_data,
                                     loss_grad_data,
                                     softmax_data,
                                     label_data,
                                     n,
                                     d / remain,
                                     remain,
                                     ignore_index);
  }
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxGradKernel(const Context& dev_ctx,
                                       const DenseTensor& label,
                                       const DenseTensor& softmax,
                                       const DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       DenseTensor* logits_grad) {
  auto dtype = label.dtype();
  if (soft_label) {
    PADDLE_ENFORCE_EQ(
        dtype,
        phi::CppTypeToDataType<T>::Type(),
        common::errors::InvalidArgument("The Input(Label) should be with the "
                                        "same data type as kernel data type."));
    CrossEntropyWithSoftmaxGradGPUKernel<T, T>(dev_ctx,
                                               label,
                                               softmax,
                                               loss_grad,
                                               soft_label,
                                               use_softmax,
                                               numeric_stable_mode,
                                               ignore_index,
                                               axis,
                                               logits_grad);
  } else {
    PD_VISIT_INTEGRAL_TYPES(
        dtype, "CrossEntropyWithSoftmaxGradGPUKernel", ([&] {
          CrossEntropyWithSoftmaxGradGPUKernel<T, data_t>(dev_ctx,
                                                          label,
                                                          softmax,
                                                          loss_grad,
                                                          soft_label,
                                                          use_softmax,
                                                          numeric_stable_mode,
                                                          ignore_index,
                                                          axis,
                                                          logits_grad);
        }));
  }
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(cross_entropy_with_softmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(cross_entropy_with_softmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(cross_entropy_with_softmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
#endif
