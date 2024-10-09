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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename LabelT>
void CrossEntropyWithSoftmaxGradCPUKernel(const CPUContext& dev_ctx,
                                          const DenseTensor& label,
                                          const DenseTensor& softmax,
                                          const DenseTensor& loss_grad,
                                          bool soft_label,
                                          bool use_softmax,
                                          bool numeric_stable_mode UNUSED,
                                          int ignore_index,
                                          int axis,
                                          DenseTensor* logits_grad) {
  const DenseTensor* out_grad = &loss_grad;
  DenseTensor* logit_grad = logits_grad;

  if (logit_grad != &softmax || !use_softmax) {
    phi::Copy(dev_ctx, softmax, dev_ctx.GetPlace(), false, logit_grad);
  }

  const int rank = logit_grad->dims().size();
  const int axis_v = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = static_cast<int>(logit_grad->dims()[axis_v]);
  PADDLE_ENFORCE_GT(
      axis_dim,
      0,
      common::errors::InvalidArgument(
          "The axis dimension should be larger than 0, but received "
          "axis dimension is %d.",
          axis_dim));

  const int n = phi::funcs::SizeToAxis(axis_v, logit_grad->dims());
  PADDLE_ENFORCE_GT(
      n,
      0,
      common::errors::InvalidArgument(
          "The size of axis should be larger than 0, but received "
          "SizeToAxis of logit_grad is %d.",
          n));

  const int d = phi::funcs::SizeFromAxis(axis_v, logit_grad->dims());
  DenseTensor logit_grad_2d(*logit_grad);
  logit_grad_2d.Resize({n, d});
  DenseTensor labels_2d(label);
  labels_2d.Resize({n, label.numel() / n});
  DenseTensor out_grad_2d(*out_grad);
  out_grad_2d.Resize({n, d / axis_dim});

  auto out_grad_mat = EigenMatrix<T>::From(out_grad_2d);
  auto logit_grad_mat = EigenMatrix<T>::From(logit_grad_2d);
  auto& place = *dev_ctx.eigen_device();

  if (!use_softmax) {
    // use_softmax step1
    if (soft_label) {
      auto lbl_mat = EigenMatrix<T>::From(labels_2d);
      logit_grad_mat.device(place) =
          (-lbl_mat / logit_grad_mat);  // for each sample ,i  is sample id
      logit_grad_mat.device(place) =
          out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, axis_dim)) *
          logit_grad_mat;
    } else {
      // use_softmax step2
      const auto* label_data = label.data<LabelT>();
      T* logit_grad_data = logit_grad->data<T>();
      const T* out_grad_data = out_grad->data<T>();
      const int remain = d / axis_dim;
      for (int i = 0; i < n; ++i) {         // for each sample_1_dim
        for (int j = 0; j < remain; j++) {  // for each sample_other_dims
          int idx = i * remain + j;  // this sample's label_idx. for 1d case,
                                     // remain=1 and j=0, so, idx = i
          auto lbl = static_cast<int64_t>(label_data[idx]);  // NOLINT
          if (lbl == ignore_index) {
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              logit_grad_data[i * d + k * remain + j] = 0;
            }
          } else {
            // only for this sample's label_idx, the label is 1, others is 0,
            // so, only compute this label_idx's class
            logit_grad_data[i * d + lbl * remain + j] =
                (-1 / logit_grad_data[i * d + lbl * remain + j]) *
                out_grad_data[idx];
            for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
              if (k !=
                  label_data[idx]) {  // label_data[idx]: this sample's label
                logit_grad_data[i * d + k * remain + j] = 0;
              }
            }
          }
        }
      }
    }
    return;
  }
  // for use_softmax=False, continue

  if (soft_label) {
    // when soft_label = True, ignore_index is not supported
    auto lbl_mat = EigenMatrix<T>::From(labels_2d);
    logit_grad_mat.device(place) =
        out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, axis_dim)) *
        (logit_grad_mat - lbl_mat);
    // for each sample, i is sample id
    // 1) compute dy/dx by p_j - y_j or P-Y, where j is class id,
    // P=logit_grad_mat[i] is all class's probs, Y=lbl_mat[i] is
    // all class's label
    // 2) compute dy * dy/dx by   Chain rule, dy=out_grad_mat[i]
    // for high dims, e.g. (n,c) or (n,d1,...,dm, c), compute grad by matrix
    // operation

  } else {
    logit_grad_mat.device(place) =
        logit_grad_mat *  // element_wise multiply
        out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, axis_dim));

    const auto* label_data = label.data<LabelT>();
    T* logit_grad_data = logit_grad->data<T>();
    const T* out_grad_data = out_grad->data<T>();
    const int remain = d / axis_dim;
    for (int i = 0; i < n; ++i) {         // for each sample_1_dim
      for (int j = 0; j < remain; j++) {  // for each sample_other_dims
        int idx = i * remain + j;  // this sample's label_idx. for 1d case,
                                   // remain=1 and j=0, so, idx = i
        auto lbl = static_cast<int64_t>(label_data[idx]);  // NOLINT
        if (lbl == ignore_index) {
          for (int k = 0; k < axis_dim; ++k) {  // for each class id's label
            logit_grad_data[i * d + k * remain + j] = 0;
          }
        } else {
          // only for this sample's label_idx, the label is 1, others is 0,
          // so, only compute this label_idx's class
          // for 1d case, remain=1 and j=0, so, [i * d + label_data[idx] *
          // remain + j] = [i * d + label_data[idx]]
          // let idx_x = i * d + label_data[idx] * remain + j,
          //   logit_grad_data[idx_x] = logit_grad_data[idx_x] -
          //   out_grad_data[idx]
          // note: logit_grad_mat = logit_grad_mat * out_grad_mat
          // so: logit_grad_data[idx_x] =  (logit_grad_data[idx_x] - 1) *
          // out_grad_data[idx]
          // means:           dy/dp * dy=   ( p - y ) * dy

          logit_grad_data[i * d + lbl * remain + j] -= out_grad_data[idx];
        }
      }
    }
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
    CrossEntropyWithSoftmaxGradCPUKernel<T, T>(dev_ctx,
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
        dtype, "CrossEntropyWithSoftmaxGradCPUKernel", ([&] {
          CrossEntropyWithSoftmaxGradCPUKernel<T, data_t>(dev_ctx,
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

PD_REGISTER_KERNEL(cross_entropy_with_softmax_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CrossEntropyWithSoftmaxGradKernel,
                   float,
                   double) {}
