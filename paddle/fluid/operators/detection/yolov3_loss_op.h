/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
static inline bool LessEqualZero(T x) {
  return x < 1e-6;
}

template <typename T>
static T SigmoidCrossEntropy(T x, T label) {
  return (x > 0 ? x : 0.0) - x * label + std::log(1.0 + std::exp(-std::abs(x)));
}

template <typename T>
static T L1Loss(T x, T y) {
  return std::abs(y - x);
}

template <typename T>
static T SigmoidCrossEntropyGrad(T x, T label) {
  return 1.0 / (1.0 + std::exp(-x)) - label;
}

template <typename T>
static T L1LossGrad(T x, T y) {
  return x > y ? 1.0 : -1.0;
}

static int GetMaskIndex(std::vector<int> mask, int val) {
  for (size_t i = 0; i < mask.size(); i++) {
    if (mask[i] == val) {
      return i;
    }
  }
  return -1;
}

template <typename T>
struct Box {
  T x, y, w, h;
};

template <typename T>
static inline T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template <typename T>
static inline Box<T> GetYoloBox(const T* x, std::vector<int> anchors, int i,
                                int j, int an_idx, int grid_size,
                                int input_size, int index, int stride,
                                float scale, float bias) {
  Box<T> b;
  b.x = (i + sigmoid<T>(x[index]) * scale + bias) / grid_size;
  b.y = (j + sigmoid<T>(x[index + stride]) * scale + bias) / grid_size;
  b.w = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] / input_size;
  b.h = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] / input_size;
  return b;
}

template <typename T>
static inline Box<T> GetGtBox(const T* gt, int batch, int max_boxes, int idx) {
  Box<T> b;
  b.x = gt[(batch * max_boxes + idx) * 4];
  b.y = gt[(batch * max_boxes + idx) * 4 + 1];
  b.w = gt[(batch * max_boxes + idx) * 4 + 2];
  b.h = gt[(batch * max_boxes + idx) * 4 + 3];
  return b;
}

template <typename T>
static inline T BoxOverlap(T c1, T w1, T c2, T w2) {
  T l1 = c1 - w1 / 2.0;
  T l2 = c2 - w2 / 2.0;
  T left = l1 > l2 ? l1 : l2;
  T r1 = c1 + w1 / 2.0;
  T r2 = c2 + w2 / 2.0;
  T right = r1 < r2 ? r1 : r2;
  return right - left;
}

template <typename T>
static inline T CalcBoxIoU(Box<T> b1, Box<T> b2) {
  T w = BoxOverlap(b1.x, b1.w, b2.x, b2.w);
  T h = BoxOverlap(b1.y, b1.h, b2.y, b2.h);
  T inter_area = (w < 0 || h < 0) ? 0.0 : w * h;
  T union_area = b1.w * b1.h + b2.w * b2.h - inter_area;
  return inter_area / union_area;
}

static inline int GetEntryIndex(int batch, int an_idx, int hw_idx, int an_num,
                                int an_stride, int stride, int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
static void CalcBoxLocationLoss(T* loss, const T* input, Box<T> gt,
                                std::vector<int> anchors, int an_idx,
                                int box_idx, int gi, int gj, int grid_size,
                                int input_size, int stride, T score) {
  T tx = gt.x * grid_size - gi;
  T ty = gt.y * grid_size - gj;
  T tw = std::log(gt.w * input_size / anchors[2 * an_idx]);
  T th = std::log(gt.h * input_size / anchors[2 * an_idx + 1]);

  T scale = (2.0 - gt.w * gt.h) * score;
  loss[0] += SigmoidCrossEntropy<T>(input[box_idx], tx) * scale;
  loss[0] += SigmoidCrossEntropy<T>(input[box_idx + stride], ty) * scale;
  loss[0] += L1Loss<T>(input[box_idx + 2 * stride], tw) * scale;
  loss[0] += L1Loss<T>(input[box_idx + 3 * stride], th) * scale;
}

template <typename T>
static void CalcBoxLocationLossGrad(T* input_grad, const T loss, const T* input,
                                    Box<T> gt, std::vector<int> anchors,
                                    int an_idx, int box_idx, int gi, int gj,
                                    int grid_size, int input_size, int stride,
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
static inline void CalcLabelLoss(T* loss, const T* input, const int index,
                                 const int label, const int class_num,
                                 const int stride, const T pos, const T neg,
                                 T score) {
  for (int i = 0; i < class_num; i++) {
    T pred = input[index + i * stride];
    loss[0] += SigmoidCrossEntropy<T>(pred, (i == label) ? pos : neg) * score;
  }
}

template <typename T>
static inline void CalcLabelLossGrad(T* input_grad, const T loss,
                                     const T* input, const int index,
                                     const int label, const int class_num,
                                     const int stride, const T pos, const T neg,
                                     T score) {
  for (int i = 0; i < class_num; i++) {
    T pred = input[index + i * stride];
    input_grad[index + i * stride] =
        SigmoidCrossEntropyGrad<T>(pred, (i == label) ? pos : neg) * score *
        loss;
  }
}

template <typename T>
static inline void CalcObjnessLoss(T* loss, const T* input, const T* objness,
                                   const int n, const int an_num, const int h,
                                   const int w, const int stride,
                                   const int an_stride) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          T obj = objness[k * w + l];
          if (obj > 1e-5) {
            // positive sample: obj = mixup score
            loss[i] += SigmoidCrossEntropy<T>(input[k * w + l], 1.0) * obj;
          } else if (obj > -0.5) {
            // negetive sample: obj = 0
            loss[i] += SigmoidCrossEntropy<T>(input[k * w + l], 0.0);
          }
        }
      }
      objness += stride;
      input += an_stride;
    }
  }
}

template <typename T>
static inline void CalcObjnessLossGrad(T* input_grad, const T* loss,
                                       const T* input, const T* objness,
                                       const int n, const int an_num,
                                       const int h, const int w,
                                       const int stride, const int an_stride) {
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

template <typename T>
static void inline GtValid(bool* valid, const T* gtbox, const int n,
                           const int b) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < b; j++) {
      if (LessEqualZero(gtbox[j * 4 + 2]) || LessEqualZero(gtbox[j * 4 + 3])) {
        valid[j] = false;
      } else {
        valid[j] = true;
      }
    }
    valid += b;
    gtbox += b * 4;
  }
}

template <typename T>
class Yolov3LossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* gt_box = ctx.Input<Tensor>("GTBox");
    auto* gt_label = ctx.Input<Tensor>("GTLabel");
    auto* gt_score = ctx.Input<Tensor>("GTScore");
    auto* loss = ctx.Output<Tensor>("Loss");
    auto* objness_mask = ctx.Output<Tensor>("ObjectnessMask");
    auto* gt_match_mask = ctx.Output<Tensor>("GTMatchMask");
    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    auto anchor_mask = ctx.Attr<std::vector<int>>("anchor_mask");
    int class_num = ctx.Attr<int>("class_num");
    float ignore_thresh = ctx.Attr<float>("ignore_thresh");
    int downsample_ratio = ctx.Attr<int>("downsample_ratio");
    bool use_label_smooth = ctx.Attr<bool>("use_label_smooth");
    float scale = ctx.Attr<float>("scale_x_y");
    float bias = -0.5 * (scale - 1.);

    const int n = input->dims()[0];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int an_num = anchors.size() / 2;
    const int mask_num = anchor_mask.size();
    const int b = gt_box->dims()[1];
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
    const T* gt_box_data = gt_box->data<T>();
    const int* gt_label_data = gt_label->data<int>();
    T* loss_data = loss->mutable_data<T>({n}, ctx.GetPlace());
    memset(loss_data, 0, loss->numel() * sizeof(T));
    T* obj_mask_data =
        objness_mask->mutable_data<T>({n, mask_num, h, w}, ctx.GetPlace());
    memset(obj_mask_data, 0, objness_mask->numel() * sizeof(T));
    int* gt_match_mask_data =
        gt_match_mask->mutable_data<int>({n, b}, ctx.GetPlace());

    const T* gt_score_data;
    Tensor gtscore;
    if (!gt_score) {
      gtscore.mutable_data<T>({n, b}, ctx.GetPlace());
      pten::funcs::SetConstant<platform::CPUDeviceContext, T>()(
          ctx.template device_context<platform::CPUDeviceContext>(), &gtscore,
          static_cast<T>(1.0));
      gt_score = &gtscore;
      gt_score_data = gtscore.data<T>();
    } else {
      gt_score_data = gt_score->data<T>();
    }

    // calc valid gt box mask, avoid calc duplicately in following code
    Tensor gt_valid_mask;
    bool* gt_valid_mask_data =
        gt_valid_mask.mutable_data<bool>({n, b}, ctx.GetPlace());
    GtValid<T>(gt_valid_mask_data, gt_box_data, n, b);

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < mask_num; j++) {
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            // each predict box find a best match gt box, if overlap is bigger
            // then ignore_thresh, ignore the objectness loss.
            int box_idx =
                GetEntryIndex(i, j, k * w + l, mask_num, an_stride, stride, 0);
            Box<T> pred =
                GetYoloBox(input_data, anchors, l, k, anchor_mask[j], h,
                           input_size, box_idx, stride, scale, bias);
            T best_iou = 0;
            for (int t = 0; t < b; t++) {
              if (!gt_valid_mask_data[i * b + t]) {
                continue;
              }
              Box<T> gt = GetGtBox(gt_box_data, i, b, t);
              T iou = CalcBoxIoU(pred, gt);
              if (iou > best_iou) {
                best_iou = iou;
              }
            }

            // If best IoU is bigger then ignore_thresh,
            // ignore the objectness loss.
            if (best_iou > ignore_thresh) {
              int obj_idx = (i * mask_num + j) * stride + k * w + l;
              obj_mask_data[obj_idx] = static_cast<T>(-1);
            }
            // all losses should be calculated if best IoU
            // is bigger then truth thresh, but currently,
            // truth thresh is an unreachable value as 1.0.
          }
        }
      }
      for (int t = 0; t < b; t++) {
        if (!gt_valid_mask_data[i * b + t]) {
          gt_match_mask_data[i * b + t] = -1;
          continue;
        }
        Box<T> gt = GetGtBox(gt_box_data, i, b, t);
        int gi = static_cast<int>(gt.x * w);
        int gj = static_cast<int>(gt.y * h);
        Box<T> gt_shift = gt;
        gt_shift.x = 0.0;
        gt_shift.y = 0.0;
        T best_iou = 0.0;
        int best_n = 0;
        // each gt box find a best match anchor box as positive sample,
        // for positive sample, all losses should be calculated, and for
        // other samples, only objectness loss is required.
        for (int an_idx = 0; an_idx < an_num; an_idx++) {
          Box<T> an_box;
          an_box.x = 0.0;
          an_box.y = 0.0;
          an_box.w = anchors[2 * an_idx] / static_cast<T>(input_size);
          an_box.h = anchors[2 * an_idx + 1] / static_cast<T>(input_size);
          float iou = CalcBoxIoU<T>(an_box, gt_shift);
          if (iou > best_iou) {
            best_iou = iou;
            best_n = an_idx;
          }
        }

        int mask_idx = GetMaskIndex(anchor_mask, best_n);
        gt_match_mask_data[i * b + t] = mask_idx;
        if (mask_idx >= 0) {
          T score = gt_score_data[i * b + t];
          int box_idx = GetEntryIndex(i, mask_idx, gj * w + gi, mask_num,
                                      an_stride, stride, 0);
          CalcBoxLocationLoss<T>(loss_data + i, input_data, gt, anchors, best_n,
                                 box_idx, gi, gj, h, input_size, stride, score);

          int obj_idx = (i * mask_num + mask_idx) * stride + gj * w + gi;
          obj_mask_data[obj_idx] = score;

          int label = gt_label_data[i * b + t];
          int label_idx = GetEntryIndex(i, mask_idx, gj * w + gi, mask_num,
                                        an_stride, stride, 5);
          CalcLabelLoss<T>(loss_data + i, input_data, label_idx, label,
                           class_num, stride, label_pos, label_neg, score);
        }
      }
    }

    CalcObjnessLoss<T>(loss_data, input_data + 4 * stride, obj_mask_data, n,
                       mask_num, h, w, stride, an_stride);
  }
};

template <typename T>
class Yolov3LossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* gt_box = ctx.Input<Tensor>("GTBox");
    auto* gt_label = ctx.Input<Tensor>("GTLabel");
    auto* gt_score = ctx.Input<Tensor>("GTScore");
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto* objness_mask = ctx.Input<Tensor>("ObjectnessMask");
    auto* gt_match_mask = ctx.Input<Tensor>("GTMatchMask");
    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    auto anchor_mask = ctx.Attr<std::vector<int>>("anchor_mask");
    int class_num = ctx.Attr<int>("class_num");
    int downsample_ratio = ctx.Attr<int>("downsample_ratio");
    bool use_label_smooth = ctx.Attr<bool>("use_label_smooth");

    const int n = input_grad->dims()[0];
    const int c = input_grad->dims()[1];
    const int h = input_grad->dims()[2];
    const int w = input_grad->dims()[3];
    const int mask_num = anchor_mask.size();
    const int b = gt_match_mask->dims()[1];
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
    const T* gt_box_data = gt_box->data<T>();
    const int* gt_label_data = gt_label->data<int>();
    const T* loss_grad_data = loss_grad->data<T>();
    const T* obj_mask_data = objness_mask->data<T>();
    const int* gt_match_mask_data = gt_match_mask->data<int>();
    T* input_grad_data =
        input_grad->mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    memset(input_grad_data, 0, input_grad->numel() * sizeof(T));

    const T* gt_score_data;
    Tensor gtscore;
    if (!gt_score) {
      gtscore.mutable_data<T>({n, b}, ctx.GetPlace());
      pten::funcs::SetConstant<platform::CPUDeviceContext, T>()(
          ctx.template device_context<platform::CPUDeviceContext>(), &gtscore,
          static_cast<T>(1.0));
      gt_score = &gtscore;
      gt_score_data = gtscore.data<T>();
    } else {
      gt_score_data = gt_score->data<T>();
    }

    for (int i = 0; i < n; i++) {
      for (int t = 0; t < b; t++) {
        int mask_idx = gt_match_mask_data[i * b + t];
        if (mask_idx >= 0) {
          T score = gt_score_data[i * b + t];
          Box<T> gt = GetGtBox(gt_box_data, i, b, t);
          int gi = static_cast<int>(gt.x * w);
          int gj = static_cast<int>(gt.y * h);

          int box_idx = GetEntryIndex(i, mask_idx, gj * w + gi, mask_num,
                                      an_stride, stride, 0);
          CalcBoxLocationLossGrad<T>(input_grad_data, loss_grad_data[i],
                                     input_data, gt, anchors,
                                     anchor_mask[mask_idx], box_idx, gi, gj, h,
                                     input_size, stride, score);

          int label = gt_label_data[i * b + t];
          int label_idx = GetEntryIndex(i, mask_idx, gj * w + gi, mask_num,
                                        an_stride, stride, 5);
          CalcLabelLossGrad<T>(input_grad_data, loss_grad_data[i], input_data,
                               label_idx, label, class_num, stride, label_pos,
                               label_neg, score);
        }
      }
    }

    CalcObjnessLossGrad<T>(input_grad_data + 4 * stride, loss_grad_data,
                           input_data + 4 * stride, obj_mask_data, n, mask_num,
                           h, w, stride, an_stride);
  }
};

}  // namespace operators
}  // namespace paddle
