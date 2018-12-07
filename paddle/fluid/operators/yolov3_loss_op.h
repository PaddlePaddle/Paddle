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

using Array5 = Eigen::DSizes<int64_t, 5>;

template <typename T>
static inline bool isZero(T x) {
  return fabs(x) < 1e-6;
}

template <typename T>
static inline void CalcMSEWithWeight(const Tensor& x, const Tensor& y,
                                     const Tensor& weight, const T loss_weight,
                                     T* loss) {
  int n = x.dims()[0];
  int stride = x.numel() / n;
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  const T* weight_data = weight.data<T>();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < stride; j++) {
      loss[i] += pow(y_data[j] - x_data[j], 2) * weight_data[j] * loss_weight;
    }
    x_data += stride;
    y_data += stride;
    weight_data += stride;
  }
}

template <typename T>
static void CalcMSEGradWithWeight(const T* loss_grad, Tensor* grad,
                                  const Tensor& x, const Tensor& y,
                                  const Tensor& weight) {
  int n = x.dims()[0];
  int stride = x.numel() / n;
  T* grad_data = grad->data<T>();
  const T* x_data = x.data<T>();
  const T* y_data = y.data<T>();
  const T* weight_data = weight.data<T>();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < stride; j++) {
      grad_data[j] =
          2.0 * weight_data[j] * (x_data[j] - y_data[j]) * loss_grad[i];
    }
    grad_data += stride;
    x_data += stride;
    y_data += stride;
    weight_data += stride;
  }
}

template <typename T>
static inline void CalcSCEWithWeight(const Tensor& x, const Tensor& label,
                                     const Tensor& weight, const T loss_weight,
                                     T* loss) {
  int n = x.dims()[0];
  int stride = x.numel() / n;
  const T* x_data = x.data<T>();
  const T* label_data = label.data<T>();
  const T* weight_data = weight.data<T>();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < stride; j++) {
      T term1 = (x_data[j] > 0) ? x_data[j] : 0;
      T term2 = x_data[j] * label_data[j];
      T term3 = std::log(1.0 + std::exp(-std::abs(x_data[j])));
      loss[i] += (term1 - term2 + term3) * weight_data[j] * loss_weight;
    }
    x_data += stride;
    label_data += stride;
    weight_data += stride;
  }
}

template <typename T>
static inline void CalcSCEGradWithWeight(const T* loss_grad, Tensor* grad,
                                         const Tensor& x, const Tensor& label,
                                         const Tensor& weight) {
  int n = x.dims()[0];
  int stride = x.numel() / n;
  T* grad_data = grad->data<T>();
  const T* x_data = x.data<T>();
  const T* label_data = label.data<T>();
  const T* weight_data = weight.data<T>();

  // LOG(ERROR) << "SCE grad start";
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < stride; j++) {
      grad_data[j] = (1.0 / (1.0 + std::exp(-x_data[j])) - label_data[j]) *
                     weight_data[j] * loss_grad[i];
      // if (j == 18) LOG(ERROR) << x_data[j] << " " << label_data[j] << " " <<
      // weight_data[j] << " " << loss_grad[i];
    }
    grad_data += stride;
    x_data += stride;
    label_data += stride;
    weight_data += stride;
  }
}

template <typename T>
static void SplitPredResult(const Tensor& input, Tensor* pred_conf,
                            Tensor* pred_class, Tensor* pred_x, Tensor* pred_y,
                            Tensor* pred_w, Tensor* pred_h,
                            const int anchor_num, const int class_num) {
  const int n = input.dims()[0];
  const int h = input.dims()[2];
  const int w = input.dims()[3];
  const int box_attr_num = 5 + class_num;

  auto input_t = EigenTensor<T, 4>::From(input);
  auto pred_conf_t = EigenTensor<T, 4>::From(*pred_conf);
  auto pred_class_t = EigenTensor<T, 5>::From(*pred_class);
  auto pred_x_t = EigenTensor<T, 4>::From(*pred_x);
  auto pred_y_t = EigenTensor<T, 4>::From(*pred_y);
  auto pred_w_t = EigenTensor<T, 4>::From(*pred_w);
  auto pred_h_t = EigenTensor<T, 4>::From(*pred_h);

  for (int i = 0; i < n; i++) {
    for (int an_idx = 0; an_idx < anchor_num; an_idx++) {
      for (int j = 0; j < h; j++) {
        for (int k = 0; k < w; k++) {
          pred_x_t(i, an_idx, j, k) = input_t(i, box_attr_num * an_idx, j, k);
          pred_y_t(i, an_idx, j, k) =
              input_t(i, box_attr_num * an_idx + 1, j, k);
          pred_w_t(i, an_idx, j, k) =
              input_t(i, box_attr_num * an_idx + 2, j, k);
          pred_h_t(i, an_idx, j, k) =
              input_t(i, box_attr_num * an_idx + 3, j, k);

          pred_conf_t(i, an_idx, j, k) =
              input_t(i, box_attr_num * an_idx + 4, j, k);

          for (int c = 0; c < class_num; c++) {
            pred_class_t(i, an_idx, j, k, c) =
                input_t(i, box_attr_num * an_idx + 5 + c, j, k);
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
static void PreProcessGTBox(const Tensor& gt_box, const Tensor& gt_label,
                            const float ignore_thresh, std::vector<int> anchors,
                            const int input_size, const int grid_size,
                            Tensor* obj_mask, Tensor* noobj_mask, Tensor* tx,
                            Tensor* ty, Tensor* tw, Tensor* th, Tensor* tweight,
                            Tensor* tconf, Tensor* tclass) {
  const int n = gt_box.dims()[0];
  const int b = gt_box.dims()[1];
  const int anchor_num = anchors.size() / 2;
  auto gt_box_t = EigenTensor<T, 3>::From(gt_box);
  auto gt_label_t = EigenTensor<int, 2>::From(gt_label);
  auto obj_mask_t = EigenTensor<T, 4>::From(*obj_mask).setConstant(0);
  auto noobj_mask_t = EigenTensor<T, 4>::From(*noobj_mask).setConstant(1);
  auto tx_t = EigenTensor<T, 4>::From(*tx).setConstant(0.0);
  auto ty_t = EigenTensor<T, 4>::From(*ty).setConstant(0.0);
  auto tw_t = EigenTensor<T, 4>::From(*tw).setConstant(0.0);
  auto th_t = EigenTensor<T, 4>::From(*th).setConstant(0.0);
  auto tweight_t = EigenTensor<T, 4>::From(*tweight).setConstant(0.0);
  auto tconf_t = EigenTensor<T, 4>::From(*tconf).setConstant(0.0);
  auto tclass_t = EigenTensor<T, 5>::From(*tclass).setConstant(0.0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < b; j++) {
      if (isZero<T>(gt_box_t(i, j, 0)) && isZero<T>(gt_box_t(i, j, 1)) &&
          isZero<T>(gt_box_t(i, j, 2)) && isZero<T>(gt_box_t(i, j, 3))) {
        continue;
      }

      int cur_label = gt_label_t(i, j);
      T gx = gt_box_t(i, j, 0) * grid_size;
      T gy = gt_box_t(i, j, 1) * grid_size;
      T gw = gt_box_t(i, j, 2) * input_size;
      T gh = gt_box_t(i, j, 3) * input_size;
      int gi = static_cast<int>(gx);
      int gj = static_cast<int>(gy);

      T max_iou = static_cast<T>(0);
      T iou;
      int best_an_index = -1;
      std::vector<T> gt_box_shape({0, 0, gw, gh});
      for (int an_idx = 0; an_idx < anchor_num; an_idx++) {
        std::vector<T> anchor_shape({0, 0, static_cast<T>(anchors[2 * an_idx]),
                                     static_cast<T>(anchors[2 * an_idx + 1])});
        iou = CalcBoxIoU<T>(gt_box_shape, anchor_shape);
        if (iou > max_iou) {
          max_iou = iou;
          best_an_index = an_idx;
        }
        if (iou > ignore_thresh) {
          noobj_mask_t(i, an_idx, gj, gi) = static_cast<T>(0.0);
        }
      }
      obj_mask_t(i, best_an_index, gj, gi) = static_cast<T>(1.0);
      noobj_mask_t(i, best_an_index, gj, gi) = static_cast<T>(0.0);
      tx_t(i, best_an_index, gj, gi) = gx - gi;
      ty_t(i, best_an_index, gj, gi) = gy - gj;
      tw_t(i, best_an_index, gj, gi) = log(gw / anchors[2 * best_an_index]);
      th_t(i, best_an_index, gj, gi) = log(gh / anchors[2 * best_an_index + 1]);
      tweight_t(i, best_an_index, gj, gi) =
          2.0 - gt_box_t(i, j, 2) * gt_box_t(i, j, 3);
      tclass_t(i, best_an_index, gj, gi, cur_label) = 1;
      tconf_t(i, best_an_index, gj, gi) = 1;
    }
  }
}

template <typename T>
static void AddAllGradToInputGrad(
    Tensor* grad, const Tensor& grad_x, const Tensor& grad_y,
    const Tensor& grad_w, const Tensor& grad_h, const Tensor& grad_conf_target,
    const Tensor& grad_conf_notarget, const Tensor& grad_class,
    const int class_num, const float loss_weight_xy, const float loss_weight_wh,
    const float loss_weight_conf_target, const float loss_weight_conf_notarget,
    const float loss_weight_class) {
  const int n = grad_x.dims()[0];
  const int an_num = grad_x.dims()[1];
  const int h = grad_x.dims()[2];
  const int w = grad_x.dims()[3];
  const int attr_num = class_num + 5;
  auto grad_t = EigenTensor<T, 4>::From(*grad).setConstant(0.0);
  auto grad_x_t = EigenTensor<T, 4>::From(grad_x);
  auto grad_y_t = EigenTensor<T, 4>::From(grad_y);
  auto grad_w_t = EigenTensor<T, 4>::From(grad_w);
  auto grad_h_t = EigenTensor<T, 4>::From(grad_h);
  auto grad_conf_target_t = EigenTensor<T, 4>::From(grad_conf_target);
  auto grad_conf_notarget_t = EigenTensor<T, 4>::From(grad_conf_notarget);
  auto grad_class_t = EigenTensor<T, 5>::From(grad_class);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          grad_t(i, j * attr_num, k, l) = grad_x_t(i, j, k, l) * loss_weight_xy;
          grad_t(i, j * attr_num + 1, k, l) =
              grad_y_t(i, j, k, l) * loss_weight_xy;
          grad_t(i, j * attr_num + 2, k, l) =
              grad_w_t(i, j, k, l) * loss_weight_wh;
          grad_t(i, j * attr_num + 3, k, l) =
              grad_h_t(i, j, k, l) * loss_weight_wh;
          grad_t(i, j * attr_num + 4, k, l) =
              grad_conf_target_t(i, j, k, l) * loss_weight_conf_target;
          grad_t(i, j * attr_num + 4, k, l) +=
              grad_conf_notarget_t(i, j, k, l) * loss_weight_conf_notarget;

          for (int c = 0; c < class_num; c++) {
            grad_t(i, j * attr_num + 5 + c, k, l) =
                grad_class_t(i, j, k, l, c) * loss_weight_class;
          }
        }
      }
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
    int class_num = ctx.Attr<int>("class_num");
    int input_size = ctx.Attr<int>("input_size");
    float ignore_thresh = ctx.Attr<float>("ignore_thresh");
    float loss_weight_xy = ctx.Attr<float>("loss_weight_xy");
    float loss_weight_wh = ctx.Attr<float>("loss_weight_wh");
    float loss_weight_conf_target = ctx.Attr<float>("loss_weight_conf_target");
    float loss_weight_conf_notarget =
        ctx.Attr<float>("loss_weight_conf_notarget");
    float loss_weight_class = ctx.Attr<float>("loss_weight_class");

    const int n = input->dims()[0];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int an_num = anchors.size() / 2;

    Tensor pred_x, pred_y, pred_w, pred_h;
    Tensor pred_conf, pred_class;
    pred_x.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_y.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_w.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_h.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_conf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_class.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    SplitPredResult<T>(*input, &pred_conf, &pred_class, &pred_x, &pred_y,
                       &pred_w, &pred_h, an_num, class_num);

    Tensor obj_mask, noobj_mask;
    Tensor tx, ty, tw, th, tweight, tconf, tclass;
    obj_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    noobj_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tx.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    ty.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tw.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    th.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tweight.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tconf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tclass.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    PreProcessGTBox<T>(*gt_box, *gt_label, ignore_thresh, anchors, input_size,
                       h, &obj_mask, &noobj_mask, &tx, &ty, &tw, &th, &tweight,
                       &tconf, &tclass);

    Tensor obj_weight;
    obj_weight.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    auto obj_weight_t = EigenTensor<T, 4>::From(obj_weight);
    auto obj_mask_t = EigenTensor<T, 4>::From(obj_mask);
    auto tweight_t = EigenTensor<T, 4>::From(tweight);
    obj_weight_t = obj_mask_t * tweight_t;

    Tensor obj_mask_expand;
    obj_mask_expand.mutable_data<T>({n, an_num, h, w, class_num},
                                    ctx.GetPlace());
    auto obj_mask_expand_t = EigenTensor<T, 5>::From(obj_mask_expand);
    obj_mask_expand_t = obj_mask_t.reshape(Array5(n, an_num, h, w, 1))
                            .broadcast(Array5(1, 1, 1, 1, class_num));

    T* loss_data = loss->mutable_data<T>({n}, ctx.GetPlace());
    memset(loss_data, 0, n * sizeof(T));
    CalcSCEWithWeight<T>(pred_x, tx, obj_weight, loss_weight_xy, loss_data);
    CalcSCEWithWeight<T>(pred_y, ty, obj_weight, loss_weight_xy, loss_data);
    CalcMSEWithWeight<T>(pred_w, tw, obj_weight, loss_weight_wh, loss_data);
    CalcMSEWithWeight<T>(pred_h, th, obj_weight, loss_weight_wh, loss_data);
    CalcSCEWithWeight<T>(pred_conf, tconf, obj_mask, loss_weight_conf_target,
                         loss_data);
    CalcSCEWithWeight<T>(pred_conf, tconf, noobj_mask,
                         loss_weight_conf_notarget, loss_data);
    CalcSCEWithWeight<T>(pred_class, tclass, obj_mask_expand, loss_weight_class,
                         loss_data);

    // loss_data[0] = (loss_weight_xy * (loss_x + loss_y) +
    //                loss_weight_wh * (loss_w + loss_h) +
    //                loss_weight_conf_target * loss_conf_target +
    //                loss_weight_conf_notarget * loss_conf_notarget +
    //                loss_weight_class * loss_class) / n;
  }
};

template <typename T>
class Yolov3LossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* gt_box = ctx.Input<Tensor>("GTBox");
    auto* gt_label = ctx.Input<Tensor>("GTLabel");
    auto anchors = ctx.Attr<std::vector<int>>("anchors");
    int class_num = ctx.Attr<int>("class_num");
    float ignore_thresh = ctx.Attr<float>("ignore_thresh");
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));
    const T* loss_grad_data = loss_grad->data<T>();
    int input_size = ctx.Attr<int>("input_size");
    float loss_weight_xy = ctx.Attr<float>("loss_weight_xy");
    float loss_weight_wh = ctx.Attr<float>("loss_weight_wh");
    float loss_weight_conf_target = ctx.Attr<float>("loss_weight_conf_target");
    float loss_weight_conf_notarget =
        ctx.Attr<float>("loss_weight_conf_notarget");
    float loss_weight_class = ctx.Attr<float>("loss_weight_class");

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int an_num = anchors.size() / 2;

    Tensor pred_x, pred_y, pred_w, pred_h;
    Tensor pred_conf, pred_class;
    pred_x.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_y.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_w.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_h.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_conf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    pred_class.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    SplitPredResult<T>(*input, &pred_conf, &pred_class, &pred_x, &pred_y,
                       &pred_w, &pred_h, an_num, class_num);

    Tensor obj_mask, noobj_mask;
    Tensor tx, ty, tw, th, tweight, tconf, tclass;
    obj_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    noobj_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tx.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    ty.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tw.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    th.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tweight.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tconf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tclass.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    PreProcessGTBox<T>(*gt_box, *gt_label, ignore_thresh, anchors, input_size,
                       h, &obj_mask, &noobj_mask, &tx, &ty, &tw, &th, &tweight,
                       &tconf, &tclass);

    Tensor obj_weight;
    obj_weight.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    auto obj_weight_t = EigenTensor<T, 4>::From(obj_weight);
    auto obj_mask_t = EigenTensor<T, 4>::From(obj_mask);
    auto tweight_t = EigenTensor<T, 4>::From(tweight);
    obj_weight_t = obj_mask_t * tweight_t;

    // LOG(ERROR) << obj_mask_t;

    Tensor obj_mask_expand;
    obj_mask_expand.mutable_data<T>({n, an_num, h, w, class_num},
                                    ctx.GetPlace());
    auto obj_mask_expand_t = EigenTensor<T, 5>::From(obj_mask_expand);
    obj_mask_expand_t = obj_mask_t.reshape(Array5(n, an_num, h, w, 1))
                            .broadcast(Array5(1, 1, 1, 1, class_num));

    Tensor grad_x, grad_y, grad_w, grad_h;
    Tensor grad_conf_target, grad_conf_notarget, grad_class;
    grad_x.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    grad_y.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    grad_w.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    grad_h.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    grad_conf_target.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    grad_conf_notarget.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    grad_class.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());
    CalcSCEGradWithWeight<T>(loss_grad_data, &grad_x, pred_x, tx, obj_weight);
    CalcSCEGradWithWeight<T>(loss_grad_data, &grad_y, pred_y, ty, obj_weight);
    CalcMSEGradWithWeight<T>(loss_grad_data, &grad_w, pred_w, tw, obj_weight);
    CalcMSEGradWithWeight<T>(loss_grad_data, &grad_h, pred_h, th, obj_weight);
    CalcSCEGradWithWeight<T>(loss_grad_data, &grad_conf_target, pred_conf,
                             tconf, obj_mask);
    CalcSCEGradWithWeight<T>(loss_grad_data, &grad_conf_notarget, pred_conf,
                             tconf, noobj_mask);
    CalcSCEGradWithWeight<T>(loss_grad_data, &grad_class, pred_class, tclass,
                             obj_mask_expand);

    input_grad->mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    AddAllGradToInputGrad<T>(input_grad, grad_x, grad_y, grad_w, grad_h,
                             grad_conf_target, grad_conf_notarget, grad_class,
                             class_num, loss_weight_xy, loss_weight_wh,
                             loss_weight_conf_target, loss_weight_conf_notarget,
                             loss_weight_class);
  }
};

}  // namespace operators
}  // namespace paddle
