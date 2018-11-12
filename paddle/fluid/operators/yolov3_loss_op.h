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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

using Array2 = Eigen::DSizes<int64_t, 2>;
using Array4 = Eigen::DSizes<int64_t, 4>;

template <typename T>
static inline bool isZero(T x) {
  return abs(x) < 1e-6;
}

template <typename T>
static inline T sigmod(T x) {
  return 1.0 / (exp(-1.0 * x) + 1.0);
}

template <typename T>
static inline T CalcMSEWithMask(const Tensor& x, const Tensor& y,
                                const Tensor& mask) {
  auto x_t = EigenVector<T>::Flatten(x);
  auto y_t = EigenVector<T>::Flatten(y);
  auto mask_t = EigenVector<T>::Flatten(mask);

  T error_sum = 0.0;
  T points = 0.0;
  for (int i = 0; i < x_t.dimensions()[0]; i++) {
    if (mask_t(i)) {
      error_sum += pow(x_t(i) - y_t(i), 2);
      points += 1;
    }
  }
  return (error_sum / points);
}

template <typename T>
static inline T CalcBCEWithMask(const Tensor& x, const Tensor& y,
                                const Tensor& mask) {
  auto x_t = EigenVector<T>::Flatten(x);
  auto y_t = EigenVector<T>::Flatten(y);
  auto mask_t = EigenVector<T>::Flatten(mask);

  T error_sum = 0.0;
  T points = 0.0;
  for (int i = 0; i < x_t.dimensions()[0]; i++) {
    if (mask_t(i)) {
      error_sum +=
          -1.0 * (y_t(i) * log(x_t(i)) + (1.0 - y_t(i)) * log(1.0 - x_t(i)));
      points += 1;
    }
  }
  return (error_sum / points);
}

template <typename T>
static void CalcPredResult(const Tensor& input, Tensor* pred_confs,
                           Tensor* pred_classes, Tensor* pred_x, Tensor* pred_y,
                           Tensor* pred_w, Tensor* pred_h,
                           std::vector<int> anchors, const int class_num,
                           const int stride) {
  const int n = input.dims()[0];
  const int c = input.dims()[1];
  const int h = input.dims()[2];
  const int w = input.dims()[3];
  const int anchor_num = anchors.size() / 2;
  const int box_attr_num = 5 + class_num;

  auto input_t = EigenTensor<T, 4>::From(input);
  // auto pred_boxes_t = EigenTensor<T, 5>::From(*pred_boxes);
  auto pred_confs_t = EigenTensor<T, 4>::From(*pred_confs);
  auto pred_classes_t = EigenTensor<T, 5>::From(*pred_classes);
  auto pred_x_t = EigenTensor<T, 4>::From(*pred_x);
  auto pred_y_t = EigenTensor<T, 4>::From(*pred_y);
  auto pred_w_t = EigenTensor<T, 4>::From(*pred_w);
  auto pred_h_t = EigenTensor<T, 4>::From(*pred_h);

  for (int i = 0; i < n; i++) {
    for (int an_idx = 0; an_idx < anchor_num; an_idx++) {
      float an_w = anchors[an_idx * 2] / stride;
      float an_h = anchors[an_idx * 2 + 1] / stride;

      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          pred_x_t(i, an_idx, j, k) =
              sigmod(input_t(i, box_attr_num * an_idx, j, k));
          pred_y_t(i, an_idx, j, k) =
              sigmod(input_t(i, box_attr_num * an_idx + 1, j, k));
          pred_w_t(i, an_idx, j, k) =
              input_t(i, box_attr_num * an_idx + 2, j, k);
          pred_h_t(i, an_idx, j, k) =
              input_t(i, box_attr_num * an_idx + 3, j, k);

          // pred_boxes_t(i, an_idx, j, k, 0) = pred_x_t(i, an_idx, j, k) + k;
          // pred_boxes_t(i, an_idx, j, k, 1) = pred_y_t(i, an_idx, j, k) + j;
          // pred_boxes_t(i, an_idx, j, k, 2) =
          //     exp(pred_w_t(i, an_idx, j, k)) * an_w;
          // pred_boxes_t(i, an_idx, j, k, 3) =
          //     exp(pred_h_t(i, an_idx, j, k)) * an_h;

          pred_confs_t(i, an_idx, j, k) =
              sigmod(input_t(i, box_attr_num * an_idx + 4, j, k));

          for (int c = 0; c < class_num; c++) {
            pred_classes_t(i, an_idx, j, k, c) =
                sigmod(input_t(i, box_attr_num * an_idx + 5 + c, j, k));
          }
        }
      }
    }
  }
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
static inline int GetPredLabel(const Tensor& pred_classes, int n,
                               int best_an_index, int gj, int gi) {
  auto pred_classes_t = EigenTensor<T, 5>::From(pred_classes);
  T score = 0.0;
  int label = -1;
  for (int i = 0; i < pred_classes.dims()[4]; i++) {
    if (pred_classes_t(n, best_an_index, gj, gi, i) > score) {
      score = pred_classes_t(n, best_an_index, gj, gi, i);
      label = i;
    }
  }
  return label;
}

template <typename T>
static void PrePorcessGTBox(const Tensor& gt_boxes, const float ignore_thresh,
                            std::vector<int> anchors, const int img_height,
                            const int grid_size, Tensor* obj_mask,
                            Tensor* noobj_mask, Tensor* tx, Tensor* ty,
                            Tensor* tw, Tensor* th, Tensor* tconf,
                            Tensor* tclass) {
  const int n = gt_boxes.dims()[0];
  const int b = gt_boxes.dims()[1];
  const int anchor_num = anchors.size() / 2;
  auto gt_boxes_t = EigenTensor<T, 3>::From(gt_boxes);
  auto obj_mask_t = EigenTensor<int, 4>::From(*obj_mask).setConstant(0);
  auto noobj_mask_t = EigenTensor<int, 4>::From(*noobj_mask).setConstant(1);
  auto tx_t = EigenTensor<T, 4>::From(*tx).setConstant(0.0);
  auto ty_t = EigenTensor<T, 4>::From(*ty).setConstant(0.0);
  auto tw_t = EigenTensor<T, 4>::From(*tw).setConstant(0.0);
  auto th_t = EigenTensor<T, 4>::From(*th).setConstant(0.0);
  auto tconf_t = EigenTensor<T, 4>::From(*tconf).setConstant(0.0);
  auto tclass_t = EigenTensor<T, 5>::From(*tclass).setConstant(0.0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < b; j++) {
      if (isZero(gt_boxes_t(i, j, 0)) && isZero(gt_boxes_t(i, j, 1)) &&
          isZero(gt_boxes_t(i, j, 2)) && isZero(gt_boxes_t(i, j, 3))) {
        continue;
      }

      int gt_label = gt_boxes_t(i, j, 0);
      T gx = gt_boxes_t(i, j, 1) * grid_size;
      T gy = gt_boxes_t(i, j, 2) * grid_size;
      T gw = gt_boxes_t(i, j, 3) * grid_size;
      T gh = gt_boxes_t(i, j, 4) * grid_size;
      int gi = static_cast<int>(gx);
      int gj = static_cast<int>(gy);

      T max_iou = static_cast<T>(-1);
      T iou;
      int best_an_index = -1;
      std::vector<T> gt_box({0, 0, gw, gh});
      for (int an_idx = 0; an_idx < anchor_num; an_idx++) {
        std::vector<T> anchor_shape({0, 0, static_cast<T>(anchors[2 * an_idx]),
                                     static_cast<T>(anchors[2 * an_idx + 1])});
        iou = CalcBoxIoU<T>(gt_box, anchor_shape);
        if (iou > max_iou) {
          max_iou = iou;
          best_an_index = an_idx;
        }
        if (iou > ignore_thresh) {
          noobj_mask_t(b, an_idx, gj, gi) = 0;
        }
      }
      obj_mask_t(b, best_an_index, gj, gi) = 1;
      noobj_mask_t(b, best_an_index, gj, gi) = 1;
      tx_t(i, best_an_index, gj, gi) = gx - gi;
      ty_t(i, best_an_index, gj, gi) = gy - gj;
      tw_t(i, best_an_index, gj, gi) = log(gw / anchors[2 * best_an_index]);
      th_t(i, best_an_index, gj, gi) = log(gh / anchors[2 * best_an_index + 1]);
      tclass_t(b, best_an_index, gj, gi, gt_label) = 1;
      tconf_t(b, best_an_index, gj, gi) = 1;
    }
  }
  noobj_mask_t = noobj_mask_t - obj_mask_t;
}

template <typename DeviceContext, typename T>
class Yolov3LossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* gt_boxes = ctx.Input<Tensor>("GTBox");
    auto* loss = ctx.Output<Tensor>("Loss");
    int img_height = ctx.Attr<int>("img_height");
    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    int class_num = ctx.Attr<int>("class_num");
    float ignore_thresh = ctx.Attr<float>("ignore_thresh");

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int an_num = anchors.size() / 2;
    const T stride = static_cast<T>(img_height) / h;

    Tensor pred_x, pred_y, pred_w, pred_h;
    Tensor pred_confs, pred_classes;
    pred_x.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_y.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_w.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_h.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_confs.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_classes.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    CalcPredResult<T>(*input, &pred_confs, &pred_classes, &pred_x, &pred_y,
                      &pred_w, &pred_h, anchors, class_num, stride);

    Tensor obj_mask, noobj_mask;
    Tensor tx, ty, tw, th, tconf, tclass;
    obj_mask.mutable_data<int>({n, an_num, h, w}, ctx.GetPlace());
    noobj_mask.mutable_data<int>({n, an_num, h, w}, ctx.GetPlace());
    tx.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    ty.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tw.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    th.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tconf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tclass.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    PrePorcessGTBox<T>(*gt_boxes, ignore_thresh, anchors, img_height, h,
                       &obj_mask, &noobj_mask, &tx, &ty, &tw, &th, &tconf,
                       &tclass);

    T loss_x = CalcMSEWithMask<T>(pred_x, tx, obj_mask);
    T loss_y = CalcMSEWithMask<T>(pred_y, ty, obj_mask);
    T loss_w = CalcMSEWithMask<T>(pred_w, tw, obj_mask);
    T loss_h = CalcMSEWithMask<T>(pred_h, th, obj_mask);
    T loss_conf_true = CalcBCEWithMask<T>(pred_confs, tconf, obj_mask);
    T loss_conf_false = CalcBCEWithMask<T>(pred_confs, tconf, noobj_mask);
    T loss_class = CalcBCEWithMask<T>(pred_classes, tclass, obj_mask);

    auto* loss_data = loss->mutable_data<T>({1}, ctx.GetPlace());
    loss_data[0] = loss_x + loss_y + loss_w + loss_h + loss_conf_true +
                   loss_conf_false + loss_class;
  }
};

template <typename DeviceContext, typename T>
class Yolov3LossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_input_t = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_output_t = ctx.Input<Tensor>(framework::GradVarName("Out"));
  }
};

}  // namespace operators
}  // namespace paddle
