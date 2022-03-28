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

#include <algorithm>
#include <vector>

#include "paddle/phi/kernels/yolov3_loss_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/yolov3_loss_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
static T SigmoidCrossEntropyGrad(T x, T label) {
  return 1.0 / (1.0 + std::exp(-x)) - label;
}

template <typename T>
static T L1LossGrad(T x, T y) {
  return x > y ? 1.0 : -1.0;
}

template <typename T>
static void CalcBoxLocationLossGrad(T* input_grad,
                                    const T loss,
                                    const T* input,
                                    Box<T> gt,
                                    std::vector<int> anchors,
                                    int an_idx,
                                    int box_idx,
                                    int gi,
                                    int gj,
                                    int grid_size,
                                    int input_size,
                                    int stride,
                                    T score) {
  T tx = gt.x * grid_size - gi;
  T ty = gt.y * grid_size - gj;
  T tw = std::log(gt.w * input_size / anchors[2 * an_idx]);
  T th = std::log(gt.h * input_size / anchors[2 * an_idx + 1]);

  T scale = (2.0 - gt.w * gt.h) * score;
  input_grad[box_idx] =
      SigmoidCrossEntropyGrad<T>(input[box_idx], tx) * scale * loss;
  input_grad[box_idx + stride] =
      SigmoidCrossEntropyGrad<T>(input[box_idx + stride], ty) * scale * loss;
  input_grad[box_idx + 2 * stride] =
      L1LossGrad<T>(input[box_idx + 2 * stride], tw) * scale * loss;
  input_grad[box_idx + 3 * stride] =
      L1LossGrad<T>(input[box_idx + 3 * stride], th) * scale * loss;
}

template <typename T>
static inline void CalcLabelLossGrad(T* input_grad,
                                     const T loss,
                                     const T* input,
                                     const int index,
                                     const int label,
                                     const int class_num,
                                     const int stride,
                                     const T pos,
                                     const T neg,
                                     T score) {
  for (int i = 0; i < class_num; i++) {
    T pred = input[index + i * stride];
    input_grad[index + i * stride] =
        SigmoidCrossEntropyGrad<T>(pred, (i == label) ? pos : neg) * score *
        loss;
  }
}

template <typename T>
static inline void CalcObjnessLossGrad(T* input_grad,
                                       const T* loss,
                                       const T* input,
                                       const T* objness,
                                       const int n,
                                       const int an_num,
                                       const int h,
                                       const int w,
                                       const int stride,
                                       const int an_stride) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          T obj = objness[k * w + l];
          if (obj > 1e-5) {
            input_grad[k * w + l] =
                SigmoidCrossEntropyGrad<T>(input[k * w + l], 1.0) * obj *
                loss[i];
          } else if (obj > -0.5) {
            input_grad[k * w + l] =
                SigmoidCrossEntropyGrad<T>(input[k * w + l], 0.0) * loss[i];
          }
        }
      }
      objness += stride;
      input += an_stride;
      input_grad += an_stride;
    }
  }
}

template <typename T, typename Context>
void Yolov3LossGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& gt_box,
                          const DenseTensor& gt_label,
                          paddle::optional<const DenseTensor&> gt_score,
                          const DenseTensor& loss_grad,
                          const DenseTensor& objectness_mask,
                          const DenseTensor& gt_match_mask,
                          const std::vector<int>& anchors,
                          const std::vector<int>& anchor_mask,
                          int class_num,
                          float ignore_thresh,
                          int downsample_ratio,
                          bool use_label_smooth,
                          float scale_x_y,
                          DenseTensor* x_grad,
                          DenseTensor* gt_box_grad,
                          DenseTensor* gt_label_grad,
                          DenseTensor* gt_score_grad) {
  auto* input = &x;
  auto input_grad = x_grad;
  auto* objness_mask = &objectness_mask;

  const int n = input_grad->dims()[0];
  const int c = input_grad->dims()[1];
  const int h = input_grad->dims()[2];
  const int w = input_grad->dims()[3];
  const int mask_num = anchor_mask.size();
  const int b = gt_match_mask.dims()[1];
  int input_size = downsample_ratio * h;

  const int stride = h * w;
  const int an_stride = (class_num + 5) * stride;

  T label_pos = 1.0;
  T label_neg = 0.0;
  if (use_label_smooth) {
    T smooth_weight = std::min(1.0 / static_cast<T>(class_num), 1.0 / 40);
    label_pos = 1.0 - smooth_weight;
    label_neg = smooth_weight;
  }

  const T* input_data = input->data<T>();
  const T* gt_box_data = gt_box.data<T>();
  const int* gt_label_data = gt_label.data<int>();
  const T* loss_grad_data = loss_grad.data<T>();
  const T* obj_mask_data = objness_mask->data<T>();
  const int* gt_match_mask_data = gt_match_mask.data<int>();
  input_grad->Resize({n, c, h, w});
  T* input_grad_data = dev_ctx.template Alloc<T>(input_grad);
  memset(input_grad_data, 0, input_grad->numel() * sizeof(T));

  const T* gt_score_data;
  DenseTensor gtscore;
  if (!(gt_score.is_initialized())) {
    gtscore.Resize({n, b});
    dev_ctx.template Alloc<T>(&gtscore);
    phi::funcs::SetConstant<Context, T>()(
        dev_ctx, &gtscore, static_cast<T>(1.0));
    gt_score_data = gtscore.data<T>();
  } else {
    gt_score_data = gt_score.get_ptr()->data<T>();
  }

  for (int i = 0; i < n; i++) {
    for (int t = 0; t < b; t++) {
      int mask_idx = gt_match_mask_data[i * b + t];
      if (mask_idx >= 0) {
        T score = gt_score_data[i * b + t];
        Box<T> gt = GetGtBox(gt_box_data, i, b, t);
        int gi = static_cast<int>(gt.x * w);
        int gj = static_cast<int>(gt.y * h);

        int box_idx = GetEntryIndex(
            i, mask_idx, gj * w + gi, mask_num, an_stride, stride, 0);
        CalcBoxLocationLossGrad<T>(input_grad_data,
                                   loss_grad_data[i],
                                   input_data,
                                   gt,
                                   anchors,
                                   anchor_mask[mask_idx],
                                   box_idx,
                                   gi,
                                   gj,
                                   h,
                                   input_size,
                                   stride,
                                   score);

        int label = gt_label_data[i * b + t];
        int label_idx = GetEntryIndex(
            i, mask_idx, gj * w + gi, mask_num, an_stride, stride, 5);
        CalcLabelLossGrad<T>(input_grad_data,
                             loss_grad_data[i],
                             input_data,
                             label_idx,
                             label,
                             class_num,
                             stride,
                             label_pos,
                             label_neg,
                             score);
      }
    }
  }

  CalcObjnessLossGrad<T>(input_grad_data + 4 * stride,
                         loss_grad_data,
                         input_data + 4 * stride,
                         obj_mask_data,
                         n,
                         mask_num,
                         h,
                         w,
                         stride,
                         an_stride);
}

}  // namespace phi

PD_REGISTER_KERNEL(yolov3_loss_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::Yolov3LossGradKernel,
                   float,
                   double) {}
