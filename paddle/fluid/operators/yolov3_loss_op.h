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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

using Array5 = Eigen::DSizes<int64_t, 5>;

template <typename T>
static inline bool isZero(T x) {
  return fabs(x) < 1e-6;
}

template <typename T>
static T CalcBoxIoU(std::vector<T> box1, std::vector<T> box2) {
  T b1_x1 = box1[0] - box1[2] / 2;
  T b1_x2 = box1[0] + box1[2] / 2;
  T b1_y1 = box1[1] - box1[3] / 2;
  T b1_y2 = box1[1] + box1[3] / 2;
  T b2_x1 = box2[0] - box2[2] / 2;
  T b2_x2 = box2[0] + box2[2] / 2;
  T b2_y1 = box2[1] - box2[3] / 2;
  T b2_y2 = box2[1] + box2[3] / 2;

  T b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);
  T b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1);

  T inter_rect_x1 = std::max(b1_x1, b2_x1);
  T inter_rect_y1 = std::max(b1_y1, b2_y1);
  T inter_rect_x2 = std::min(b1_x2, b2_x2);
  T inter_rect_y2 = std::min(b1_y2, b2_y2);
  T inter_area = std::max(inter_rect_x2 - inter_rect_x1, static_cast<T>(0.0)) *
                 std::max(inter_rect_y2 - inter_rect_y1, static_cast<T>(0.0));

  return inter_area / (b1_area + b2_area - inter_area);
}

template <typename T>
static void PreProcessGTBox(const Tensor& gt_box, const Tensor& gt_label,
                            const float ignore_thresh, std::vector<int> anchors,
                            const int input_size, const int grid_size,
                            Tensor* conf_mask, Tensor* obj_mask, Tensor* tx,
                            Tensor* ty, Tensor* tw, Tensor* th, Tensor* tweight,
                            Tensor* tconf, Tensor* tclass) {
  const int n = gt_box.dims()[0];
  const int b = gt_box.dims()[1];
  const int an_num = anchors.size() / 2;
  const int h = tclass->dims()[2];
  const int w = tclass->dims()[3];
  const int class_num = tclass->dims()[4];

  const T* gt_box_data = gt_box.data<T>();
  const int* gt_label_data = gt_label.data<int>();
  T* conf_mask_data = conf_mask->data<T>();
  T* obj_mask_data = obj_mask->data<T>();
  T* tx_data = tx->data<T>();
  T* ty_data = ty->data<T>();
  T* tw_data = tw->data<T>();
  T* th_data = th->data<T>();
  T* tweight_data = tweight->data<T>();
  T* tconf_data = tconf->data<T>();
  T* tclass_data = tclass->data<T>();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < b; j++) {
      int box_idx = (i * b + j) * 4;
      if (isZero<T>(gt_box_data[box_idx + 2]) &&
          isZero<T>(gt_box_data[box_idx + 3])) {
        continue;
      }

      int cur_label = gt_label_data[i * b + j];
      T gx = gt_box_data[box_idx] * grid_size;
      T gy = gt_box_data[box_idx + 1] * grid_size;
      T gw = gt_box_data[box_idx + 2] * input_size;
      T gh = gt_box_data[box_idx + 3] * input_size;
      int gi = static_cast<int>(gx);
      int gj = static_cast<int>(gy);

      T max_iou = static_cast<T>(0);
      T iou;
      int best_an_index = -1;
      std::vector<T> gt_box_shape({0, 0, gw, gh});
      for (int an_idx = 0; an_idx < an_num; an_idx++) {
        std::vector<T> anchor_shape({0, 0, static_cast<T>(anchors[2 * an_idx]),
                                     static_cast<T>(anchors[2 * an_idx + 1])});
        iou = CalcBoxIoU<T>(gt_box_shape, anchor_shape);
        if (iou > max_iou) {
          max_iou = iou;
          best_an_index = an_idx;
        }
        if (iou > ignore_thresh) {
          int conf_idx = ((i * an_num + an_idx) * h + gj) * w + gi;
          conf_mask_data[conf_idx] = static_cast<T>(0.0);
        }
      }

      int obj_idx = ((i * an_num + best_an_index) * h + gj) * w + gi;
      conf_mask_data[obj_idx] = static_cast<T>(1.0);
      obj_mask_data[obj_idx] = static_cast<T>(1.0);
      tx_data[obj_idx] = gx - gi;
      ty_data[obj_idx] = gy - gj;
      tw_data[obj_idx] = log(gw / anchors[2 * best_an_index]);
      th_data[obj_idx] = log(gh / anchors[2 * best_an_index + 1]);
      tweight_data[obj_idx] =
          2.0 - gt_box_data[box_idx + 2] * gt_box_data[box_idx + 3];
      tconf_data[obj_idx] = static_cast<T>(1.0);
      tclass_data[obj_idx * class_num + cur_label] = static_cast<T>(1.0);
    }
  }
}

template <typename T>
static T SCE(T x, T label) {
  return (x > 0 ? x : 0.0) - x * label + std::log(1.0 + std::exp(-std::abs(x)));
}

template <typename T>
static T L1Loss(T x, T y) {
  return std::abs(y - x);
}

template <typename T>
static T SCEGrad(T x, T label) {
  return 1.0 / (1.0 + std::exp(-x)) - label;
}

template <typename T>
static T L1LossGrad(T x, T y) {
  return x > y ? 1.0 : -1.0;
}

template <typename T>
static void CalcSCE(T* loss_data, const T* input, const T* target,
                    const T* weight, const T* mask, const int n,
                    const int an_num, const int grid_num, const int class_num,
                    const int num) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < grid_num; k++) {
        int sub_idx = k * num;
        for (int l = 0; l < num; l++) {
          loss_data[i] += SCE<T>(input[l * grid_num + k], target[sub_idx + l]) *
                          weight[k] * mask[k];
        }
      }
      input += (class_num + 5) * grid_num;
      target += grid_num * num;
      weight += grid_num;
      mask += grid_num;
    }
  }
}

template <typename T>
static void CalcSCEGrad(T* input_grad, const T* loss_grad, const T* input,
                        const T* target, const T* weight, const T* mask,
                        const int n, const int an_num, const int grid_num,
                        const int class_num, const int num) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < grid_num; k++) {
        int sub_idx = k * num;
        for (int l = 0; l < num; l++) {
          input_grad[l * grid_num + k] =
              SCEGrad<T>(input[l * grid_num + k], target[sub_idx + l]) *
              weight[k] * mask[k] * loss_grad[i];
        }
      }
      input_grad += (class_num + 5) * grid_num;
      input += (class_num + 5) * grid_num;
      target += grid_num * num;
      weight += grid_num;
      mask += grid_num;
    }
  }
}

template <typename T>
static void CalcL1Loss(T* loss_data, const T* input, const T* target,
                       const T* weight, const T* mask, const int n,
                       const int an_num, const int grid_num,
                       const int class_num) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < grid_num; k++) {
        loss_data[i] += L1Loss<T>(input[k], target[k]) * weight[k] * mask[k];
      }
      input += (class_num + 5) * grid_num;
      target += grid_num;
      weight += grid_num;
      mask += grid_num;
    }
  }
}

template <typename T>
static void CalcL1LossGrad(T* input_grad, const T* loss_grad, const T* input,
                           const T* target, const T* weight, const T* mask,
                           const int n, const int an_num, const int grid_num,
                           const int class_num) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < grid_num; k++) {
        input_grad[k] = L1LossGrad<T>(input[k], target[k]) * weight[k] *
                        mask[k] * loss_grad[i];
      }
      input_grad += (class_num + 5) * grid_num;
      input += (class_num + 5) * grid_num;
      target += grid_num;
      weight += grid_num;
      mask += grid_num;
    }
  }
}

template <typename T>
static void CalcYolov3Loss(T* loss_data, const Tensor& input, const Tensor& tx,
                           const Tensor& ty, const Tensor& tw, const Tensor& th,
                           const Tensor& tweight, const Tensor& tconf,
                           const Tensor& tclass, const Tensor& conf_mask,
                           const Tensor& obj_mask) {
  const T* input_data = input.data<T>();
  const T* tx_data = tx.data<T>();
  const T* ty_data = ty.data<T>();
  const T* tw_data = tw.data<T>();
  const T* th_data = th.data<T>();
  const T* tweight_data = tweight.data<T>();
  const T* tconf_data = tconf.data<T>();
  const T* tclass_data = tclass.data<T>();
  const T* conf_mask_data = conf_mask.data<T>();
  const T* obj_mask_data = obj_mask.data<T>();

  const int n = tclass.dims()[0];
  const int an_num = tclass.dims()[1];
  const int h = tclass.dims()[2];
  const int w = tclass.dims()[3];
  const int class_num = tclass.dims()[4];
  const int grid_num = h * w;

  CalcSCE<T>(loss_data, input_data, tx_data, tweight_data, obj_mask_data, n,
             an_num, grid_num, class_num, 1);
  CalcSCE<T>(loss_data, input_data + grid_num, ty_data, tweight_data,
             obj_mask_data, n, an_num, grid_num, class_num, 1);
  CalcL1Loss<T>(loss_data, input_data + 2 * grid_num, tw_data, tweight_data,
                obj_mask_data, n, an_num, grid_num, class_num);
  CalcL1Loss<T>(loss_data, input_data + 3 * grid_num, th_data, tweight_data,
                obj_mask_data, n, an_num, grid_num, class_num);
  CalcSCE<T>(loss_data, input_data + 4 * grid_num, tconf_data, conf_mask_data,
             conf_mask_data, n, an_num, grid_num, class_num, 1);
  CalcSCE<T>(loss_data, input_data + 5 * grid_num, tclass_data, obj_mask_data,
             obj_mask_data, n, an_num, grid_num, class_num, class_num);
}

template <typename T>
static void CalcYolov3LossGrad(T* input_grad_data, const Tensor& loss_grad,
                               const Tensor& input, const Tensor& tx,
                               const Tensor& ty, const Tensor& tw,
                               const Tensor& th, const Tensor& tweight,
                               const Tensor& tconf, const Tensor& tclass,
                               const Tensor& conf_mask,
                               const Tensor& obj_mask) {
  const T* loss_grad_data = loss_grad.data<T>();
  const T* input_data = input.data<T>();
  const T* tx_data = tx.data<T>();
  const T* ty_data = ty.data<T>();
  const T* tw_data = tw.data<T>();
  const T* th_data = th.data<T>();
  const T* tweight_data = tweight.data<T>();
  const T* tconf_data = tconf.data<T>();
  const T* tclass_data = tclass.data<T>();
  const T* conf_mask_data = conf_mask.data<T>();
  const T* obj_mask_data = obj_mask.data<T>();

  const int n = tclass.dims()[0];
  const int an_num = tclass.dims()[1];
  const int h = tclass.dims()[2];
  const int w = tclass.dims()[3];
  const int class_num = tclass.dims()[4];
  const int grid_num = h * w;

  CalcSCEGrad<T>(input_grad_data, loss_grad_data, input_data, tx_data,
                 tweight_data, obj_mask_data, n, an_num, grid_num, class_num,
                 1);
  CalcSCEGrad<T>(input_grad_data + grid_num, loss_grad_data,
                 input_data + grid_num, ty_data, tweight_data, obj_mask_data, n,
                 an_num, grid_num, class_num, 1);
  CalcL1LossGrad<T>(input_grad_data + 2 * grid_num, loss_grad_data,
                    input_data + 2 * grid_num, tw_data, tweight_data,
                    obj_mask_data, n, an_num, grid_num, class_num);
  CalcL1LossGrad<T>(input_grad_data + 3 * grid_num, loss_grad_data,
                    input_data + 3 * grid_num, th_data, tweight_data,
                    obj_mask_data, n, an_num, grid_num, class_num);
  CalcSCEGrad<T>(input_grad_data + 4 * grid_num, loss_grad_data,
                 input_data + 4 * grid_num, tconf_data, conf_mask_data,
                 conf_mask_data, n, an_num, grid_num, class_num, 1);
  CalcSCEGrad<T>(input_grad_data + 5 * grid_num, loss_grad_data,
                 input_data + 5 * grid_num, tclass_data, obj_mask_data,
                 obj_mask_data, n, an_num, grid_num, class_num, class_num);
}

static int mask_index(std::vector<int> mask, int val) {
  for (int i = 0; i < mask.size(); i++) {
    if (mask[i] == val) {
      return i;
    }
  }
  return -1;
}

template <typename T>
struct Box {
  float x, y, w, h;
};

template <typename T>
static inline T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template <typename T>
static inline void sigmoid_arrray(T* arr, int len) {
  for (int i = 0; i < len; i++) {
    arr[i] = sigmoid(arr[i]);
  }
}

template <typename T>
static inline Box<T> get_yolo_box(const T* x, std::vector<int> anchors, int i,
                                  int j, int an_idx, int grid_size,
                                  int input_size, int index, int stride) {
  Box<T> b;
  b.x = (i + sigmoid<T>(x[index])) / grid_size;
  b.y = (j + sigmoid<T>(x[index + stride])) / grid_size;
  b.w = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] / input_size;
  b.h = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] / input_size;
  return b;
}

template <typename T>
static inline Box<T> get_gt_box(const T* gt, int batch, int max_boxes,
                                int idx) {
  Box<T> b;
  b.x = gt[(batch * max_boxes + idx) * 4];
  b.y = gt[(batch * max_boxes + idx) * 4 + 1];
  b.w = gt[(batch * max_boxes + idx) * 4 + 2];
  b.h = gt[(batch * max_boxes + idx) * 4 + 3];
  return b;
}

template <typename T>
static inline T overlap(T c1, T w1, T c2, T w2) {
  T l1 = c1 - w1 / 2.0;
  T l2 = c2 - w2 / 2.0;
  T left = l1 > l2 ? l1 : l2;
  T r1 = c1 + w1 / 2.0;
  T r2 = c2 + w2 / 2.0;
  T right = r1 < r2 ? r1 : r2;
  return right - left;
}

template <typename T>
static inline T box_iou(Box<T> b1, Box<T> b2) {
  T w = overlap(b1.x, b1.w, b2.x, b2.w);
  T h = overlap(b1.y, b1.h, b2.y, b2.h);
  T inter_area = (w < 0 || h < 0) ? 0.0 : w * h;
  T union_area = b1.w * b1.h + b2.w * b2.h - inter_area;
  return inter_area / union_area;
}

static inline int entry_index(int batch, int an_idx, int hw_idx, int an_num,
                              int an_stride, int stride, int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
static void CalcBoxLocationLoss(T* loss, const T* input, Box<T> gt,
                                std::vector<int> anchors, int an_idx,
                                int box_idx, int gi, int gj, int grid_size,
                                int input_size, int stride) {
  T tx = gt.x * grid_size - gi;
  T ty = gt.y * grid_size - gj;
  T tw = std::log(gt.w * input_size / anchors[2 * an_idx]);
  T th = std::log(gt.h * input_size / anchors[2 * an_idx + 1]);

  T scale = 2.0 - gt.w * gt.h;
  loss[0] += SCE<T>(input[box_idx], tx) * scale;
  loss[0] += SCE<T>(input[box_idx + stride], ty) * scale;
  loss[0] += L1Loss<T>(input[box_idx + 2 * stride], tw) * scale;
  loss[0] += L1Loss<T>(input[box_idx + 3 * stride], th) * scale;
}

template <typename T>
static void CalcBoxLocationLossGrad(T* input_grad, const T loss, const T* input,
                                    Box<T> gt, std::vector<int> anchors,
                                    int an_idx, int box_idx, int gi, int gj,
                                    int grid_size, int input_size, int stride) {
  T tx = gt.x * grid_size - gi;
  T ty = gt.y * grid_size - gj;
  T tw = std::log(gt.w * input_size / anchors[2 * an_idx]);
  T th = std::log(gt.h * input_size / anchors[2 * an_idx + 1]);

  T scale = 2.0 - gt.w * gt.h;
  input_grad[box_idx] = SCEGrad<T>(input[box_idx], tx) * scale * loss;
  input_grad[box_idx + stride] =
      SCEGrad<T>(input[box_idx + stride], ty) * scale * loss;
  input_grad[box_idx + 2 * stride] =
      L1LossGrad<T>(input[box_idx + 2 * stride], tw) * scale * loss;
  input_grad[box_idx + 3 * stride] =
      L1LossGrad<T>(input[box_idx + 3 * stride], th) * scale * loss;
}

template <typename T>
static inline void CalcLabelLoss(T* loss, const T* input, const int index,
                                 const int label, const int class_num,
                                 const int stride) {
  for (int i = 0; i < class_num; i++) {
    loss[0] += SCE<T>(input[index + i * stride], (i == label) ? 1.0 : 0.0);
  }
}

template <typename T>
static inline void CalcLabelLossGrad(T* input_grad, const T loss,
                                     const T* input, const int index,
                                     const int label, const int class_num,
                                     const int stride) {
  for (int i = 0; i < class_num; i++) {
    input_grad[index + i * stride] =
        SCEGrad<T>(input[index + i * stride], (i == label) ? 1.0 : 0.0) * loss;
  }
}

template <typename T>
static inline void CalcObjnessLoss(T* loss, const T* input, const int* objness,
                                   const int n, const int an_num, const int h,
                                   const int w, const int stride,
                                   const int an_stride) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          int obj = objness[k * w + l];
          if (obj >= 0) {
            loss[i] += SCE<T>(input[k * w + l], static_cast<T>(obj));
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
                                       const T* input, const int* objness,
                                       const int n, const int an_num,
                                       const int h, const int w,
                                       const int stride, const int an_stride) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          int obj = objness[k * w + l];
          if (obj >= 0) {
            input_grad[k * w + l] =
                SCEGrad<T>(input[k * w + l], static_cast<T>(obj)) * loss[i];
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
class Yolov3LossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* gt_box = ctx.Input<Tensor>("GTBox");
    auto* gt_label = ctx.Input<Tensor>("GTLabel");
    auto* loss = ctx.Output<Tensor>("Loss");
    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    auto anchor_mask = ctx.Attr<std::vector<int>>("anchor_mask");
    int class_num = ctx.Attr<int>("class_num");
    float ignore_thresh = ctx.Attr<float>("ignore_thresh");
    int downsample = ctx.Attr<int>("downsample");

    const int n = input->dims()[0];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int an_num = anchors.size() / 2;
    const int mask_num = anchor_mask.size();
    const int b = gt_box->dims()[1];
    int input_size = downsample * h;

    const T* input_data = input->data<T>();
    const T* gt_box_data = gt_box->data<T>();
    const int* gt_label_data = gt_label->data<int>();
    T* loss_data = loss->mutable_data<T>({n}, ctx.GetPlace());
    memset(loss_data, 0, n * sizeof(int));

    Tensor objness;
    int* objness_data =
        objness.mutable_data<int>({n, mask_num, h, w}, ctx.GetPlace());
    memset(objness_data, 0, objness.numel() * sizeof(int));

    const int stride = h * w;
    const int an_stride = (class_num + 5) * stride;

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < mask_num; j++) {
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            int box_idx =
                entry_index(i, j, k * w + l, mask_num, an_stride, stride, 0);
            Box<T> pred =
                get_yolo_box(input_data, anchors, l, k, anchor_mask[j], h,
                             input_size, box_idx, stride);
            T best_iou = 0;
            // int best_t = 0;
            for (int t = 0; t < b; t++) {
              if (isZero<T>(gt_box_data[i * b * 4 + t * 4]) &&
                  isZero<T>(gt_box_data[i * b * 4 + t * 4 + 1])) {
                continue;
              }
              Box<T> gt = get_gt_box(gt_box_data, i, b, t);
              T iou = box_iou(pred, gt);
              if (iou > best_iou) {
                best_iou = iou;
                // best_t = t;
              }
            }

            if (best_iou > ignore_thresh) {
              int obj_idx = (i * mask_num + j) * stride + k * w + l;
              objness_data[obj_idx] = -1;
            }
          }
        }
      }
      for (int t = 0; t < b; t++) {
        if (isZero<T>(gt_box_data[i * b * 4 + t * 4]) &&
            isZero<T>(gt_box_data[i * b * 4 + t * 4 + 1])) {
          continue;
        }
        Box<T> gt = get_gt_box(gt_box_data, i, b, t);
        int gi = static_cast<int>(gt.x * w);
        int gj = static_cast<int>(gt.y * h);
        Box<T> gt_shift = gt;
        gt_shift.x = 0.0;
        gt_shift.y = 0.0;
        T best_iou = 0.0;
        int best_n = 0;
        for (int an_idx = 0; an_idx < an_num; an_idx++) {
          Box<T> an_box;
          an_box.x = 0.0;
          an_box.y = 0.0;
          an_box.w = anchors[2 * an_idx] / static_cast<T>(input_size);
          an_box.h = anchors[2 * an_idx + 1] / static_cast<T>(input_size);
          float iou = box_iou<T>(an_box, gt_shift);
          // TO DO: iou > 0.5 ?
          if (iou > best_iou) {
            best_iou = iou;
            best_n = an_idx;
          }
        }

        int mask_idx = mask_index(anchor_mask, best_n);
        if (mask_idx >= 0) {
          int box_idx = entry_index(i, mask_idx, gj * w + gi, mask_num,
                                    an_stride, stride, 0);
          CalcBoxLocationLoss<T>(loss_data + i, input_data, gt, anchors, best_n,
                                 box_idx, gi, gj, h, input_size, stride);

          int obj_idx = (i * mask_num + mask_idx) * stride + gj * w + gi;
          objness_data[obj_idx] = 1;

          int label = gt_label_data[i * b + t];
          int label_idx = entry_index(i, mask_idx, gj * w + gi, mask_num,
                                      an_stride, stride, 5);
          CalcLabelLoss<T>(loss_data + i, input_data, label_idx, label,
                           class_num, stride);
        }
      }
    }

    CalcObjnessLoss<T>(loss_data, input_data + 4 * stride, objness_data, n,
                       mask_num, h, w, stride, an_stride);

    // Tensor conf_mask, obj_mask;
    // Tensor tx, ty, tw, th, tweight, tconf, tclass;
    // conf_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // obj_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tx.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // ty.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tw.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // th.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tweight.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tconf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tclass.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    //
    // math::SetConstant<platform::CPUDeviceContext, T> constant;
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    //          &conf_mask, static_cast<T>(1.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    //          &obj_mask, static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(), &tx,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(), &ty,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(), &tw,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(), &th,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    //          &tweight, static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    // &tconf,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    // &tclass,
    //          static_cast<T>(0.0));
    //
    // PreProcessGTBox<T>(*gt_box, *gt_label, ignore_thresh, anchors,
    // input_size,
    //                    h, &conf_mask, &obj_mask, &tx, &ty, &tw, &th,
    //                    &tweight,
    //                    &tconf, &tclass);
    //
    // T* loss_data = loss->mutable_data<T>({n}, ctx.GetPlace());
    // memset(loss_data, 0, n * sizeof(T));
    // CalcYolov3Loss<T>(loss_data, *input, tx, ty, tw, th, tweight, tconf,
    // tclass,
    //                   conf_mask, obj_mask);
  }
};

template <typename T>
class Yolov3LossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* gt_box = ctx.Input<Tensor>("GTBox");
    auto* gt_label = ctx.Input<Tensor>("GTLabel");
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    auto anchor_mask = ctx.Attr<std::vector<int>>("anchor_mask");
    int class_num = ctx.Attr<int>("class_num");
    float ignore_thresh = ctx.Attr<float>("ignore_thresh");
    int downsample = ctx.Attr<int>("downsample");

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int an_num = anchors.size() / 2;
    const int mask_num = anchor_mask.size();
    const int b = gt_box->dims()[1];
    int input_size = downsample * h;

    const T* input_data = input->data<T>();
    const T* gt_box_data = gt_box->data<T>();
    const int* gt_label_data = gt_label->data<int>();
    const T* loss_grad_data = loss_grad->data<T>();
    T* input_grad_data =
        input_grad->mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    memset(input_grad_data, 0, input_grad->numel() * sizeof(T));

    Tensor objness;
    int* objness_data =
        objness.mutable_data<int>({n, mask_num, h, w}, ctx.GetPlace());
    memset(objness_data, 0, objness.numel() * sizeof(int));

    const int stride = h * w;
    const int an_stride = (class_num + 5) * stride;

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < mask_num; j++) {
        for (int k = 0; k < h; k++) {
          for (int l = 0; l < w; l++) {
            int box_idx =
                entry_index(i, j, k * w + l, mask_num, an_stride, stride, 0);
            Box<T> pred =
                get_yolo_box(input_data, anchors, l, k, anchor_mask[j], h,
                             input_size, box_idx, stride);
            T best_iou = 0;
            // int best_t = 0;
            for (int t = 0; t < b; t++) {
              if (isZero<T>(gt_box_data[i * b * 4 + t * 4]) &&
                  isZero<T>(gt_box_data[i * b * 4 + t * 4 + 1])) {
                continue;
              }
              Box<T> gt = get_gt_box(gt_box_data, i, b, t);
              T iou = box_iou(pred, gt);
              if (iou > best_iou) {
                best_iou = iou;
                // best_t = t;
              }
            }

            if (best_iou > ignore_thresh) {
              int obj_idx = (i * mask_num + j) * stride + k * w + l;
              objness_data[obj_idx] = -1;
            }
          }
        }
      }
      for (int t = 0; t < b; t++) {
        if (isZero<T>(gt_box_data[i * b * 4 + t * 4]) &&
            isZero<T>(gt_box_data[i * b * 4 + t * 4 + 1])) {
          continue;
        }
        Box<T> gt = get_gt_box(gt_box_data, i, b, t);
        int gi = static_cast<int>(gt.x * w);
        int gj = static_cast<int>(gt.y * h);
        Box<T> gt_shift = gt;
        gt_shift.x = 0.0;
        gt_shift.y = 0.0;
        T best_iou = 0.0;
        int best_n = 0;
        for (int an_idx = 0; an_idx < an_num; an_idx++) {
          Box<T> an_box;
          an_box.x = 0.0;
          an_box.y = 0.0;
          an_box.w = anchors[2 * an_idx] / static_cast<T>(input_size);
          an_box.h = anchors[2 * an_idx + 1] / static_cast<T>(input_size);
          float iou = box_iou<T>(an_box, gt_shift);
          // TO DO: iou > 0.5 ?
          if (iou > best_iou) {
            best_iou = iou;
            best_n = an_idx;
          }
        }

        int mask_idx = mask_index(anchor_mask, best_n);
        if (mask_idx >= 0) {
          int box_idx = entry_index(i, mask_idx, gj * w + gi, mask_num,
                                    an_stride, stride, 0);
          CalcBoxLocationLossGrad<T>(input_grad_data, loss_grad_data[i],
                                     input_data, gt, anchors, best_n, box_idx,
                                     gi, gj, h, input_size, stride);

          int obj_idx = (i * mask_num + mask_idx) * stride + gj * w + gi;
          objness_data[obj_idx] = 1;

          int label = gt_label_data[i * b + t];
          int label_idx = entry_index(i, mask_idx, gj * w + gi, mask_num,
                                      an_stride, stride, 5);
          CalcLabelLossGrad<T>(input_grad_data, loss_grad_data[i], input_data,
                               label_idx, label, class_num, stride);
        }
      }
    }

    CalcObjnessLossGrad<T>(input_grad_data + 4 * stride, loss_grad_data,
                           input_data + 4 * stride, objness_data, n, mask_num,
                           h, w, stride, an_stride);

    // const int n = input->dims()[0];
    // const int c = input->dims()[1];
    // const int h = input->dims()[2];
    // const int w = input->dims()[3];
    // const int an_num = anchors.size() / 2;
    //
    // Tensor conf_mask, obj_mask;
    // Tensor tx, ty, tw, th, tweight, tconf, tclass;
    // conf_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // obj_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tx.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // ty.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tw.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // th.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tweight.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tconf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    // tclass.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    //
    // math::SetConstant<platform::CPUDeviceContext, T> constant;
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    //          &conf_mask, static_cast<T>(1.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    //          &obj_mask, static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(), &tx,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(), &ty,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(), &tw,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(), &th,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    //          &tweight, static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    // &tconf,
    //          static_cast<T>(0.0));
    // constant(ctx.template device_context<platform::CPUDeviceContext>(),
    // &tclass,
    //          static_cast<T>(0.0));
    //
    // PreProcessGTBox<T>(*gt_box, *gt_label, ignore_thresh, anchors,
    // input_size,
    //                    h, &conf_mask, &obj_mask, &tx, &ty, &tw, &th,
    //                    &tweight,
    //                    &tconf, &tclass);
    //
    // T* input_grad_data =
    //     input_grad->mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    // CalcYolov3LossGrad<T>(input_grad_data, *loss_grad, *input, tx, ty, tw,
    // th,
    //                       tweight, tconf, tclass, conf_mask, obj_mask);
  }
};

}  // namespace operators
}  // namespace paddle
