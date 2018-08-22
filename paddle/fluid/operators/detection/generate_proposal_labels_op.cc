/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

struct DyDataTypeVisitor {
  const platform::Place place_;
  LoDTensor* in_;
  DyDataTypeVisitor(const platform::Place& place, LoDTensor* in)
      : place_(place), in_(in) {}

  template <typename T>
  T* operator()() {
    auto* p = in_->mutable_data<T>(place_);
    return p;
  }
};

struct AppendRoisFunctor {
  LoDTensor* out_;
  int64_t offset_;
  Tensor* to_add_;

  AppendRoisFunctor(LoDTensor* out, int64_t offset, Tensor* to_add)
      : out_(out), offset_(offset), to_add_(to_add) {}

  template <typename T>
  void operator()() const {
    auto* out_data = out_->data<T>();
    auto* to_add_data = to_add_->data<T>();
    memcpy(out_data + offset_, to_add_data, to_add_->numel() * sizeof(T));
  }
};

void PrintTensor(Tensor t) {
  auto et = framework::EigenTensor<float, 2>::From(t);
  int r = t.dims()[0];
  int c = t.dims()[1];
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << et(i, j) << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void PrintVector(Tensor v) {
  auto et = framework::EigenTensor<int, 1>::From(v);
  int r = v.dims()[0];
  for (int i = 0; i < r; ++i) {
    std::cout << et(i) << ", ";
  }
  std::cout << std::endl;
  std::cout << std::endl;
}

void PrintFlag(char c) { std::cout << c << c << c << std::endl; }

class GenerateProposalLabelsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("RpnRois"),
                   "Input(RpnRois) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("GtClasses"),
                   "Input(GtClasses) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("GtBoxes"),
                   "Input(GtBoxes) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("ImScales"),
                   "Input(ImScales) shouldn't be null.");

    PADDLE_ENFORCE(ctx->HasOutput("Rois"),
                   "Output(Rois) of RpnTargetAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("LabelsInt32"),
        "Output(LabelsInt32) of RpnTargetAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("BboxTargets"),
        "Output(BboxTargets) of RpnTargetAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("BboxInsideWeights"),
        "Output(BboxInsideWeights) of RpnTargetAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("BboxOutsideWeights"),
        "Output(BboxOutsideWeights) of RpnTargetAssignOp should not be null");

    auto rpn_rois_dims = ctx->GetInputDim("RpnRois");
    auto gt_classes_dims = ctx->GetInputDim("GtClasses");
    auto gt_boxes_dims = ctx->GetInputDim("GtBoxes");
    auto im_scales_dims = ctx->GetInputDim("ImScales");

    PADDLE_ENFORCE_EQ(rpn_rois_dims.size(), 2,
                      "The rank of Input(RpnRois) must be 2.");
    PADDLE_ENFORCE_EQ(gt_classes_dims.size(), 1,
                      "The rank of Input(GtClasses) must be 1.");
    PADDLE_ENFORCE_EQ(gt_boxes_dims.size(), 2,
                      "The rank of Input(GtBoxes) must be 2.");
    PADDLE_ENFORCE_EQ(im_scales_dims.size(), 1,
                      "The rank of Input(ImScales) must be 1.");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("RpnRois"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
void Concat(const Tensor& in_tensor_a, const Tensor& in_tensor_b,
            Tensor* out_tensor) {
  int axis = 0;
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(platform::CPUPlace());

  size_t output_offset = 0;
  auto in_stride_a = framework::stride_numel(in_tensor_a.dims());
  auto in_stride_b = framework::stride_numel(in_tensor_b.dims());
  auto out_stride = framework::stride_numel(out_tensor->dims());
  StridedNumelCopyWithAxis<T>(
      dev_ctx, axis, out_tensor->data<T>() + output_offset, out_stride,
      in_tensor_a.data<T>(), in_stride_a, in_stride_a[axis]);
  output_offset += in_stride_a[axis];
  StridedNumelCopyWithAxis<T>(
      dev_ctx, axis, out_tensor->data<T>() + output_offset, out_stride,
      in_tensor_b.data<T>(), in_stride_b, in_stride_b[axis]);
}

template <typename T>
void FillConstant(Tensor* tensor, T value) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(platform::CPUPlace());
  math::set_constant(dev_ctx, tensor, value);
}

template <typename T>
void Gather(const Tensor& in_tensor, const Tensor& index, Tensor* out_tensor) {
  auto in_tensor_data = in_tensor.data<T>();
  auto index_data = index.data<int>();
  auto out_tensor_data = out_tensor->data<T>();
  int data_num = index.dims()[0];
  int stride = out_tensor->numel() / data_num;
  for (int64_t i = 0; i < data_num; ++i) {
    int idx = index_data[i];
    std::copy(in_tensor_data + (idx * stride),
              in_tensor_data + ((idx + 1) * stride),
              out_tensor_data + (i * stride));
  }
}

template <typename T>
void BboxOverlaps(const Tensor& r_boxes, const Tensor& c_boxes,
                  Tensor* overlaps) {
  auto r_boxes_et = framework::EigenTensor<T, 2>::From(r_boxes);
  auto c_boxes_et = framework::EigenTensor<T, 2>::From(c_boxes);
  auto overlaps_et = framework::EigenTensor<T, 2>::From(*overlaps);
  int r_num = r_boxes.dims()[0];
  int c_num = c_boxes.dims()[0];
  auto zero = static_cast<T>(0.0);
  T r_box_area, c_box_area, x_min, y_min, x_max, y_max, inter_w, inter_h,
      inter_area;
  for (int i = 0; i < r_num; ++i) {
    r_box_area = (r_boxes_et(i, 2) - r_boxes_et(i, 0) + 1) *
                 (r_boxes_et(i, 3) - r_boxes_et(i, 1) + 1);
    for (int j = 0; j < c_num; ++j) {
      c_box_area = (c_boxes_et(j, 2) - c_boxes_et(j, 0) + 1) *
                   (c_boxes_et(j, 3) - c_boxes_et(j, 1) + 1);
      x_min = std::max(r_boxes_et(i, 0), c_boxes_et(j, 0));
      y_min = std::max(r_boxes_et(i, 1), c_boxes_et(j, 1));
      x_max = std::min(r_boxes_et(i, 2), c_boxes_et(j, 2));
      y_max = std::min(r_boxes_et(i, 3), c_boxes_et(j, 3));
      inter_w = std::max(x_max - x_min + 1, zero);
      inter_h = std::max(y_max - y_min + 1, zero);
      inter_area = inter_w * inter_h;
      overlaps_et(i, j) = inter_area / (r_box_area + c_box_area - inter_area);
    }
  }
}

template <typename T>
void BoxToDelta(int box_num, const Tensor& ex_boxes, const Tensor& gt_boxes,
                const std::vector<float>& weights, Tensor* box_delta) {
  auto ex_boxes_et = framework::EigenTensor<T, 2>::From(ex_boxes);
  auto gt_boxes_et = framework::EigenTensor<T, 2>::From(gt_boxes);
  auto box_delta_et = framework::EigenTensor<T, 2>::From(*box_delta);
  T ex_w, ex_h, ex_ctr_x, ex_ctr_y, gt_w, gt_h, gt_ctr_x, gt_ctr_y;
  for (int64_t i = 0; i < box_num; ++i) {
    ex_w = ex_boxes_et(i, 2) - ex_boxes_et(i, 0) + 1;
    ex_h = ex_boxes_et(i, 3) - ex_boxes_et(i, 1) + 1;
    ex_ctr_x = ex_boxes_et(i, 0) + 0.5 * ex_w;
    ex_ctr_y = ex_boxes_et(i, 1) + 0.5 * ex_h;

    gt_w = gt_boxes_et(i, 2) - gt_boxes_et(i, 0) + 1;
    gt_h = gt_boxes_et(i, 3) - gt_boxes_et(i, 1) + 1;
    gt_ctr_x = gt_boxes_et(i, 0) + 0.5 * gt_w;
    gt_ctr_y = gt_boxes_et(i, 1) + 0.5 * gt_h;

    box_delta_et(i, 0) = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0];
    box_delta_et(i, 1) = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1];
    box_delta_et(i, 2) = log(gt_w / ex_w) / ex_w / weights[2];
    box_delta_et(i, 3) = log(gt_h / ex_h) / ex_h / weights[3];
  }
}

template <typename T>
std::vector<std::vector<int>> SampleFgBgGt(
    const framework::ExecutionContext& context, Tensor* iou,
    const int batch_size_per_im, const float fg_fraction, const float fg_thresh,
    const float bg_thresh_hi, const float bg_thresh_lo,
    std::minstd_rand engine) {
  std::vector<int> fg_inds;
  std::vector<int> bg_inds;
  std::vector<int> gt_inds;
  T* proposal_to_gt_overlaps = iou->mutable_data<T>(context.GetPlace());
  int64_t row = iou->dims()[0];
  int64_t col = iou->dims()[1];

  // Follow the Faster RCNN's implementation
  for (int64_t i = 0; i < row; ++i) {
    const T* v = proposal_to_gt_overlaps + i * col;
    T max_overlap = *std::max_element(v, v + col);
    if (max_overlap > fg_thresh) {
      for (int64_t j = 0; j < col; ++j) {
        T val = proposal_to_gt_overlaps[i * col + j];
        if (val == max_overlap) {
          fg_inds.emplace_back(i);
          gt_inds.emplace_back(j);
        }
      }
    } else {
      if ((max_overlap >= bg_thresh_lo) && (max_overlap < bg_thresh_hi)) {
        bg_inds.emplace_back(i);
      }
    }
  }

  // Reservoir Sampling
  int fg_rois_per_im = std::floor(batch_size_per_im * fg_fraction);
  int fg_rois_this_image = fg_inds.size();
  int fg_rois_per_this_image = std::min(fg_rois_per_im, fg_rois_this_image);
  std::uniform_real_distribution<float> uniform(0, 1);
  const int64_t fg_size = static_cast<int64_t>(fg_inds.size());
  if (fg_size > fg_rois_per_this_image) {
    for (int64_t i = fg_rois_per_this_image; i < fg_size; ++i) {
      int rng_ind = std::floor(uniform(engine) * i);
      if (rng_ind < fg_rois_per_this_image) {
        std::iter_swap(fg_inds.begin() + rng_ind, fg_inds.begin() + i);
        std::iter_swap(gt_inds.begin() + rng_ind, gt_inds.begin() + i);
      }
    }
  }
  std::vector<int> new_fg_inds(fg_inds.begin(),
                               fg_inds.begin() + fg_rois_per_this_image);
  std::vector<int> new_gt_inds(gt_inds.begin(),
                               gt_inds.begin() + fg_rois_per_this_image);

  int bg_rois_per_image = batch_size_per_im - fg_rois_per_this_image;
  int bg_rois_this_image = bg_inds.size();
  int bg_rois_per_this_image = std::min(bg_rois_per_image, bg_rois_this_image);
  const int64_t bg_size = static_cast<int64_t>(bg_inds.size());
  if (bg_size > bg_rois_per_this_image) {
    for (int64_t i = bg_rois_per_this_image; i < bg_size; ++i) {
      int rng_ind = std::floor(uniform(engine) * i);
      if (rng_ind < fg_rois_per_this_image)
        std::iter_swap(bg_inds.begin() + rng_ind, bg_inds.begin() + i);
    }
  }
  std::vector<int> new_bg_inds(bg_inds.begin(),
                               bg_inds.begin() + bg_rois_per_this_image);
  std::vector<std::vector<int>> res;
  res.emplace_back(new_fg_inds);
  res.emplace_back(new_bg_inds);
  res.emplace_back(new_gt_inds);
  return res;
}

template <typename T>
void GatherBoxesLabels(const framework::ExecutionContext& context,
                       const Tensor& boxes, const Tensor& gt_boxes,
                       const Tensor& gt_classes,
                       const std::vector<int>& fg_inds,
                       const std::vector<int>& bg_inds,
                       const std::vector<int>& gt_inds, Tensor* sampled_boxes,
                       Tensor* sampled_labels, Tensor* sampled_gts) {
  int fg_num = fg_inds.size();
  int bg_num = bg_inds.size();
  int gt_num = fg_num + bg_num;
  Tensor fg_inds_t, bg_inds_t, gt_box_inds_t, gt_label_inds_t;
  fg_inds_t.Resize({fg_num});
  bg_inds_t.Resize({bg_num});
  gt_box_inds_t.Resize({gt_num});
  gt_label_inds_t.Resize({fg_num});
  int* fg_inds_data = fg_inds_t.mutable_data<int>(context.GetPlace());
  int* bg_inds_data = bg_inds_t.mutable_data<int>(context.GetPlace());
  int* gt_box_inds_data = gt_box_inds_t.mutable_data<int>(context.GetPlace());
  int* gt_label_inds_data =
      gt_label_inds_t.mutable_data<int>(context.GetPlace());
  std::copy(fg_inds.begin(), fg_inds.end(), fg_inds_data);
  std::copy(bg_inds.begin(), bg_inds.end(), bg_inds_data);
  std::copy(gt_inds.begin(), gt_inds.end(), gt_box_inds_data);
  std::copy(gt_inds.begin(), gt_inds.end(), gt_label_inds_data);

  Tensor fg_boxes;
  fg_boxes.Resize({fg_num, 4});
  fg_boxes.mutable_data(context.GetPlace(), boxes.type());
  Gather<T>(boxes, fg_inds_t, &fg_boxes);
  Tensor bg_boxes;
  bg_boxes.Resize({bg_num, 4});
  bg_boxes.mutable_data(context.GetPlace(), boxes.type());
  Gather<T>(boxes, bg_inds_t, &bg_boxes);
  Concat<T>(fg_boxes, bg_boxes, sampled_boxes);
  Gather<T>(gt_boxes, gt_box_inds_t, sampled_gts);
  Tensor fg_labels;
  fg_labels.Resize({fg_num});
  fg_labels.mutable_data(context.GetPlace(), gt_classes.type());
  Gather<int>(gt_classes, gt_label_inds_t, &fg_labels);
  Tensor bg_labels;
  bg_labels.Resize({bg_num});
  bg_labels.mutable_data(context.GetPlace(), gt_classes.type());
  framework::AttributeMap attrs;
  FillConstant(&bg_labels, 0);
  Concat<int>(fg_labels, bg_labels, sampled_labels);
}

template <typename T>
std::vector<Tensor> SampleRoisForOneImage(
    const framework::ExecutionContext& context, Tensor* rpn_rois,
    Tensor* gt_classes, Tensor* gt_boxes, Tensor* im_scale,
    const int batch_size_per_im, const float fg_fraction, const float fg_thresh,
    const float bg_thresh_hi, const float bg_thresh_lo,
    const std::vector<float> bbox_reg_weights, const int class_nums,
    std::minstd_rand engine) {
  // Roidb
  auto rpn_rois_et = framework::EigenTensor<T, 2>::From(*rpn_rois);
  auto im_scale_data = im_scale->data<T>()[0];
  rpn_rois_et = rpn_rois_et / (im_scale_data);

  Tensor boxes;
  int proposals_num = (gt_boxes->numel() / 4) + (rpn_rois->numel() / 4);
  boxes.Resize({proposals_num, 4});
  boxes.mutable_data(context.GetPlace(), rpn_rois->type());
  Concat<T>(*gt_boxes, *rpn_rois, &boxes);

  // Overlaps
  Tensor proposal_to_gt_overlaps;
  proposal_to_gt_overlaps.Resize({proposals_num, gt_boxes->numel() / 4});
  proposal_to_gt_overlaps.mutable_data(context.GetPlace(), rpn_rois->type());
  BboxOverlaps<T>(boxes, *gt_boxes, &proposal_to_gt_overlaps);

  // Generate proposal index
  std::vector<std::vector<int>> fg_bg_gt = SampleFgBgGt<T>(
      context, &proposal_to_gt_overlaps, batch_size_per_im, fg_fraction,
      fg_thresh, bg_thresh_hi, bg_thresh_lo, engine);
  std::vector<int> fg_inds = fg_bg_gt[0];
  std::vector<int> bg_inds = fg_bg_gt[1];
  std::vector<int> gt_inds = fg_bg_gt[2];

  // Gather boxes and labels
  Tensor sampled_boxes;
  Tensor sampled_labels;
  Tensor sampled_gts;
  int boxes_num = fg_inds.size() + bg_inds.size();
  sampled_boxes.Resize({boxes_num, 4});
  sampled_labels.Resize({boxes_num});
  sampled_gts.Resize({boxes_num, 4});
  sampled_boxes.mutable_data(context.GetPlace(), rpn_rois->type());
  sampled_labels.mutable_data(context.GetPlace(), gt_classes->type());
  sampled_gts.mutable_data(context.GetPlace(), rpn_rois->type());
  GatherBoxesLabels<T>(context, boxes, *gt_boxes, *gt_classes, fg_inds, bg_inds,
                       gt_inds, &sampled_boxes, &sampled_labels, &sampled_gts);

  // Compute targets
  Tensor bbox_targets_single;
  bbox_targets_single.Resize({boxes_num, 4});
  bbox_targets_single.mutable_data(context.GetPlace(), sampled_boxes.type());
  BoxToDelta<T>(boxes_num, sampled_boxes, sampled_gts, bbox_reg_weights,
                &bbox_targets_single);

  // Scale rois
  Tensor sampled_rois;
  sampled_rois.Resize(sampled_boxes.dims());
  sampled_rois.mutable_data(context.GetPlace(), sampled_boxes.type());
  auto sampled_rois_et = framework::EigenTensor<T, 2>::From(sampled_rois);
  auto sampled_boxes_et = framework::EigenTensor<T, 2>::From(sampled_boxes);
  sampled_rois_et = sampled_boxes_et * im_scale_data;

  // Expand box targets
  Tensor bbox_targets, bbox_inside_weights, bbox_outside_weights;
  bbox_targets.Resize({boxes_num, 4 * class_nums});
  bbox_inside_weights.Resize({boxes_num, 4 * class_nums});
  bbox_outside_weights.Resize({boxes_num, 4 * class_nums});
  bbox_targets.mutable_data(context.GetPlace(), rpn_rois->type());
  bbox_inside_weights.mutable_data(context.GetPlace(), rpn_rois->type());
  bbox_outside_weights.mutable_data(context.GetPlace(), rpn_rois->type());
  FillConstant<T>(&bbox_targets, 0.0);
  FillConstant<T>(&bbox_inside_weights, 0.0);
  FillConstant<T>(&bbox_outside_weights, 0.0);
  auto bbox_targets_et = framework::EigenTensor<T, 2>::From(bbox_targets);
  auto bbox_inside_weights_et =
      framework::EigenTensor<T, 2>::From(bbox_inside_weights);
  auto bbox_outside_weights_et =
      framework::EigenTensor<T, 2>::From(bbox_outside_weights);
  auto sampled_labels_et = framework::EigenTensor<int, 1>::From(sampled_labels);
  auto bbox_targets_single_et =
      framework::EigenTensor<T, 2>::From(bbox_targets_single);
  for (int64_t i = 0; i < boxes_num; ++i) {
    int label = sampled_labels_et(i);
    if (label > 0) {
      bbox_targets_et(i, 4 * label) = bbox_targets_single_et(4 * i);
      bbox_targets_et(i, 4 * label + 1) = bbox_targets_single_et(4 * i + 1);
      bbox_targets_et(i, 4 * label + 2) = bbox_targets_single_et(4 * i + 2);
      bbox_targets_et(i, 4 * label + 3) = bbox_targets_single_et(4 * i + 3);
      bbox_inside_weights_et(i, 4 * label) = 1;
      bbox_inside_weights_et(i, 4 * label + 1) = 1;
      bbox_inside_weights_et(i, 4 * label + 2) = 1;
      bbox_inside_weights_et(i, 4 * label + 3) = 1;
      bbox_outside_weights_et(i, 4 * label) = 1;
      bbox_outside_weights_et(i, 4 * label + 1) = 1;
      bbox_outside_weights_et(i, 4 * label + 2) = 1;
      bbox_outside_weights_et(i, 4 * label + 3) = 1;
    }
  }
  std::vector<Tensor> res;
  res.emplace_back(sampled_rois);
  res.emplace_back(sampled_labels);
  res.emplace_back(bbox_targets);
  res.emplace_back(bbox_inside_weights);
  res.emplace_back(bbox_outside_weights);
  return res;
}

template <typename T>
class GenerateProposalLabelsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* rpn_rois = context.Input<LoDTensor>("RpnRois");
    auto* gt_classes = context.Input<LoDTensor>("GtClasses");
    auto* gt_boxes = context.Input<LoDTensor>("GtBoxes");
    auto* im_scales = context.Input<LoDTensor>("ImScales");

    auto* rois = context.Output<LoDTensor>("Rois");
    auto* labels_int32 = context.Output<LoDTensor>("LabelsInt32");
    auto* bbox_targets = context.Output<LoDTensor>("BboxTargets");
    auto* bbox_inside_weights = context.Output<LoDTensor>("BboxInsideWeights");
    auto* bbox_outside_weights =
        context.Output<LoDTensor>("BboxOutsideWeights");

    int batch_size_per_im = context.Attr<int>("batch_size_per_im");
    float fg_fraction = context.Attr<float>("fg_fraction");
    float fg_thresh = context.Attr<float>("fg_thresh");
    float bg_thresh_hi = context.Attr<float>("bg_thresh_hi");
    float bg_thresh_lo = context.Attr<float>("bg_thresh_lo");
    std::vector<float> bbox_reg_weights =
        context.Attr<std::vector<float>>("bbox_reg_weights");
    int class_nums = context.Attr<int>("class_nums");

    int64_t n = rpn_rois->lod().size() == 0UL
                    ? 1
                    : static_cast<int64_t>(rpn_rois->lod().back().size() - 1);
    if (rpn_rois->lod().size()) {
      PADDLE_ENFORCE_EQ(rpn_rois->lod().size(), 1UL,
                        "GenerateProposalLabelsOp needs 1 level of LoD");
      PADDLE_ENFORCE_EQ(gt_classes->lod().size(), 1UL,
                        "GenerateProposalLabelsOp needs 1 level of LoD");
      PADDLE_ENFORCE_EQ(gt_boxes->lod().size(), 1UL,
                        "GenerateProposalLabelsOp needs 1 level of LoD");
    }
    rois->Resize({n * batch_size_per_im, 4});
    labels_int32->Resize({n * batch_size_per_im});
    bbox_targets->Resize({n * batch_size_per_im, 4 * class_nums});
    bbox_inside_weights->Resize({n * batch_size_per_im, 4 * class_nums});
    bbox_outside_weights->Resize({n * batch_size_per_im, 4 * class_nums});
    framework::VisitDataType(framework::ToDataType(rpn_rois->type()),
                             DyDataTypeVisitor(context.GetPlace(), rois));
    framework::VisitDataType(
        framework::ToDataType(gt_classes->type()),
        DyDataTypeVisitor(context.GetPlace(), labels_int32));
    framework::VisitDataType(
        framework::ToDataType(rpn_rois->type()),
        DyDataTypeVisitor(context.GetPlace(), bbox_targets));
    framework::VisitDataType(
        framework::ToDataType(rpn_rois->type()),
        DyDataTypeVisitor(context.GetPlace(), bbox_inside_weights));
    framework::VisitDataType(
        framework::ToDataType(rpn_rois->type()),
        DyDataTypeVisitor(context.GetPlace(), bbox_outside_weights));

    std::random_device rnd;
    std::minstd_rand engine;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
    engine.seed(seed);

    framework::LoD lod;
    std::vector<size_t> lod0(1, 0);

    int64_t num_rois = 0;
    if (n == 1) {
      Tensor rpn_rois_slice = rpn_rois->Slice(0, rpn_rois->dims()[0]);
      Tensor gt_classes_slice = gt_classes->Slice(0, gt_classes->dims()[0]);
      Tensor gt_boxes_slice = gt_boxes->Slice(0, gt_boxes->dims()[0]);
      Tensor im_scales_slice = im_scales->Slice(0, 1);
      std::vector<Tensor> tensor_output = SampleRoisForOneImage<T>(
          context, &rpn_rois_slice, &gt_classes_slice, &gt_boxes_slice,
          &im_scales_slice, batch_size_per_im, fg_fraction, fg_thresh,
          bg_thresh_hi, bg_thresh_lo, bbox_reg_weights, class_nums, engine);
      Tensor sampled_rois = tensor_output[0];
      Tensor sampled_labels_int32 = tensor_output[1];
      Tensor sampled_bbox_targets = tensor_output[2];
      Tensor sampled_bbox_inside_weights = tensor_output[3];
      Tensor sampled_bbox_outside_weights = tensor_output[4];

      framework::VisitDataType(
          framework::ToDataType(rois->type()),
          AppendRoisFunctor(rois, 4 * num_rois, &sampled_rois));
      framework::VisitDataType(
          framework::ToDataType(labels_int32->type()),
          AppendRoisFunctor(labels_int32, num_rois, &sampled_labels_int32));
      framework::VisitDataType(
          framework::ToDataType(bbox_targets->type()),
          AppendRoisFunctor(bbox_targets, 4 * num_rois * class_nums,
                            &sampled_bbox_targets));
      framework::VisitDataType(
          framework::ToDataType(bbox_inside_weights->type()),
          AppendRoisFunctor(bbox_inside_weights, 4 * num_rois * class_nums,
                            &sampled_bbox_inside_weights));
      framework::VisitDataType(
          framework::ToDataType(bbox_outside_weights->type()),
          AppendRoisFunctor(bbox_outside_weights, 4 * num_rois * class_nums,
                            &sampled_bbox_outside_weights));

      num_rois += sampled_rois.dims()[0];
      lod0.emplace_back(num_rois);
    } else {
      auto rpn_rois_lod = rpn_rois->lod().back();
      auto gt_classes_lod = gt_classes->lod().back();
      auto gt_boxes_lod = gt_boxes->lod().back();
      for (size_t i = 0; i < rpn_rois_lod.size(); ++i) {
        Tensor rpn_rois_slice =
            rpn_rois->Slice(rpn_rois_lod[i], rpn_rois_lod[i + 1]);
        Tensor gt_classes_slice =
            gt_classes->Slice(gt_classes_lod[i], gt_classes_lod[i + 1]);
        Tensor gt_boxes_slice =
            gt_boxes->Slice(gt_boxes_lod[i], gt_boxes_lod[i + 1]);
        Tensor im_scales_slice = im_scales->Slice(i, i + 1);
        std::vector<Tensor> tensor_output = SampleRoisForOneImage<T>(
            context, &rpn_rois_slice, &gt_classes_slice, &gt_boxes_slice,
            &im_scales_slice, batch_size_per_im, fg_fraction, fg_thresh,
            bg_thresh_hi, bg_thresh_lo, bbox_reg_weights, class_nums, engine);
        Tensor sampled_rois = tensor_output[0];
        Tensor sampled_labels_int32 = tensor_output[1];
        Tensor sampled_bbox_targets = tensor_output[2];
        Tensor sampled_bbox_inside_weights = tensor_output[3];
        Tensor sampled_bbox_outside_weights = tensor_output[4];
        framework::VisitDataType(
            framework::ToDataType(rois->type()),
            AppendRoisFunctor(rois, 4 * num_rois, &sampled_rois));
        framework::VisitDataType(
            framework::ToDataType(labels_int32->type()),
            AppendRoisFunctor(labels_int32, num_rois, &sampled_labels_int32));
        framework::VisitDataType(
            framework::ToDataType(bbox_targets->type()),
            AppendRoisFunctor(bbox_targets, 4 * num_rois * class_nums,
                              &sampled_bbox_targets));
        framework::VisitDataType(
            framework::ToDataType(bbox_inside_weights->type()),
            AppendRoisFunctor(bbox_inside_weights, 4 * num_rois * class_nums,
                              &sampled_bbox_inside_weights));
        framework::VisitDataType(
            framework::ToDataType(bbox_outside_weights->type()),
            AppendRoisFunctor(bbox_outside_weights, 4 * num_rois * class_nums,
                              &sampled_bbox_outside_weights));

        num_rois += sampled_rois.dims()[0];
        lod0.emplace_back(num_rois);
      }
    }

    lod.emplace_back(lod0);
    rois->set_lod(lod);
    labels_int32->set_lod(lod);
    bbox_targets->set_lod(lod);
    bbox_inside_weights->set_lod(lod);
    bbox_outside_weights->set_lod(lod);
    rois->Resize({num_rois, 4});
    labels_int32->Resize({num_rois});
    bbox_targets->Resize({num_rois, 4 * class_nums});
    bbox_inside_weights->Resize({num_rois, 4 * class_nums});
    bbox_outside_weights->Resize({num_rois, 4 * class_nums});
  }
};

class GenerateProposalLabelsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("RpnRois", "RpnRois.");
    AddInput("GtClasses", "GtClasses.");
    AddInput("GtBoxes", "GtBoxes.");
    AddInput("ImScales", "ImScales.");

    AddOutput("Rois", "Rois.");
    AddOutput("LabelsInt32", "LabelsInt32.");
    AddOutput("BboxTargets", "BboxTargets.");
    AddOutput("BboxInsideWeights", "BboxInsideWeights.");
    AddOutput("BboxOutsideWeights", "BboxOutsideWeights.");

    AddAttr<int>("batch_size_per_im", "batch_size_per_im");
    AddAttr<float>("fg_fraction", "fg_fraction");
    AddAttr<float>("fg_thresh", "fg_thresh");
    AddAttr<float>("bg_thresh_hi", "bg_thresh_hi");
    AddAttr<float>("bg_thresh_lo", "bg_thresh_lo");
    AddAttr<std::vector<float>>("bbox_reg_weights", "bbox_reg_weights");
    AddAttr<int>("class_nums", "class_nums");
    AddAttr<bool>("fix_seed", "fix_seed").SetDefault(false);
    AddAttr<int>("seed", "seed").SetDefault(0);

    AddComment(R"DOC(
Generate Proposals Labels Operator.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(generate_proposal_labels, ops::GenerateProposalLabelsOp,
                  ops::GenerateProposalLabelsOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(generate_proposal_labels,
                       ops::GenerateProposalLabelsKernel<float>,
                       ops::GenerateProposalLabelsKernel<double>);
