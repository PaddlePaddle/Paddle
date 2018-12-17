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

  // T l = 0.0;
  CalcSCE<T>(loss_data, input_data, tx_data, tweight_data, obj_mask_data, n,
             an_num, grid_num, class_num, 1);
  CalcSCE<T>(loss_data, input_data + grid_num, ty_data, tweight_data,
             obj_mask_data, n, an_num, grid_num, class_num, 1);
  // LOG(ERROR) << "C++ xy: " << loss_data[0] - l;
  // l = loss_data[0];
  CalcL1Loss<T>(loss_data, input_data + 2 * grid_num, tw_data, tweight_data,
                obj_mask_data, n, an_num, grid_num, class_num);
  CalcL1Loss<T>(loss_data, input_data + 3 * grid_num, th_data, tweight_data,
                obj_mask_data, n, an_num, grid_num, class_num);
  // LOG(ERROR) << "C++ wh: " << loss_data[0] - l;
  // l = loss_data[0];
  CalcSCE<T>(loss_data, input_data + 4 * grid_num, tconf_data, conf_mask_data,
             conf_mask_data, n, an_num, grid_num, class_num, 1);
  // LOG(ERROR) << "C++ conf: " << loss_data[0] - l;
  // l = loss_data[0];
  CalcSCE<T>(loss_data, input_data + 5 * grid_num, tclass_data, obj_mask_data,
             obj_mask_data, n, an_num, grid_num, class_num, class_num);
  // LOG(ERROR) << "C++ class: " << loss_data[0] - l;
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

template <typename DeviceContext, typename T>
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

    const int n = input->dims()[0];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int an_num = anchors.size() / 2;

    Tensor conf_mask, obj_mask;
    Tensor tx, ty, tw, th, tweight, tconf, tclass;
    conf_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    obj_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tx.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    ty.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tw.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    th.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tweight.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tconf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tclass.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());

    math::SetConstant<DeviceContext, T> constant;
    constant(ctx.template device_context<DeviceContext>(), &conf_mask,
             static_cast<T>(1.0));
    constant(ctx.template device_context<DeviceContext>(), &obj_mask,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tx,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &ty,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tw,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &th,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tweight,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tconf,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tclass,
             static_cast<T>(0.0));

    PreProcessGTBox<T>(*gt_box, *gt_label, ignore_thresh, anchors, input_size,
                       h, &conf_mask, &obj_mask, &tx, &ty, &tw, &th, &tweight,
                       &tconf, &tclass);

    T* loss_data = loss->mutable_data<T>({n}, ctx.GetPlace());
    memset(loss_data, 0, n * sizeof(T));
    CalcYolov3Loss<T>(loss_data, *input, tx, ty, tw, th, tweight, tconf, tclass,
                      conf_mask, obj_mask);
  }
};

template <typename DeviceContext, typename T>
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
    int input_size = ctx.Attr<int>("input_size");

    const int n = input->dims()[0];
    const int c = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];
    const int an_num = anchors.size() / 2;

    Tensor conf_mask, obj_mask;
    Tensor tx, ty, tw, th, tweight, tconf, tclass;
    conf_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    obj_mask.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tx.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    ty.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tw.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    th.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tweight.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tconf.mutable_data<T>({n, an_num, h, w}, ctx.GetPlace());
    tclass.mutable_data<T>({n, an_num, h, w, class_num}, ctx.GetPlace());

    math::SetConstant<DeviceContext, T> constant;
    constant(ctx.template device_context<DeviceContext>(), &conf_mask,
             static_cast<T>(1.0));
    constant(ctx.template device_context<DeviceContext>(), &obj_mask,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tx,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &ty,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tw,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &th,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tweight,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tconf,
             static_cast<T>(0.0));
    constant(ctx.template device_context<DeviceContext>(), &tclass,
             static_cast<T>(0.0));

    PreProcessGTBox<T>(*gt_box, *gt_label, ignore_thresh, anchors, input_size,
                       h, &conf_mask, &obj_mask, &tx, &ty, &tw, &th, &tweight,
                       &tconf, &tclass);

    T* input_grad_data =
        input_grad->mutable_data<T>({n, c, h, w}, ctx.GetPlace());
    CalcYolov3LossGrad<T>(input_grad_data, *loss_grad, *input, tx, ty, tw, th,
                          tweight, tconf, tclass, conf_mask, obj_mask);
  }
};

}  // namespace operators
}  // namespace paddle
