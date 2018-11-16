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
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/reduce_sum_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
const int kBoxDim = 4;

template <typename T>
void AppendMask(LoDTensor* out, int64_t offset, Tensor* to_add) {
  auto* out_data = out->data<T>();
  auto* to_add_data = to_add->data<T>();
  memcpy(out_data + offset, to_add_data, to_add->numel() * sizeof(T));
}

class GenerateMaskLabelsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("ImInfo"), "Input(ImInfo) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("GtClasses"),
                   "Input(GtClasses) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("IsCrowd"),
                   "Input(IsCrowd) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("GtSegms"),
                   "Input(GtSegms) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Rois"), "Input(Rois) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("LabelsInt32"),
                   "Input(LabelsInt32) shouldn't be null.");

    PADDLE_ENFORCE(
        ctx->HasOutput("MaskRois"),
        "Output(MaskRois) of GenerateMaskLabelsOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("RoiHasMaskInt32"),
        "Output(RoiHasMaskInt32) of GenerateMaskLabelsOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("MaskInt32"),
        "Output(MaskInt32) of GenerateMaskLabelsOp should not be null");

    auto im_info_dims = ctx->GetInputDim("ImInfo");
    auto gt_classes_dims = ctx->GetInputDim("GtClasses");
    auto is_crowd_dims = ctx->GetInputDim("IsCrowd");
    auto gt_segms_dims = ctx->GetInputDim("GtSegms");
    auto rois_dims = ctx->GetInputDim("Rois");
    auto labels_int32_dims = ctx->GetInputDim("LabelsInt32");

    PADDLE_ENFORCE_EQ(im_info_dims.size(), 2,
                      "The rank of Input(ImInfo) must be 2.");

    int num_classes = ctx->Attrs().Get<int>("num_classes");
    int resolution = ctx->Attrs().Get<int>("resolution");

    ctx->SetOutputDim("MaskRois", {-1, 4});
    ctx->SetOutputDim("RoiHasMaskInt32", {-1, 1});
    ctx->SetOutputDim("MaskInt32", {-1, num_classes * resolution * resolution});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("Rois"));
    return framework::OpKernelType(data_type, platform::CPUPlace());
  }
};

template <typename T>
static inline Tensor MasksToBoxes(const platform::CPUDeviceContext& context,
                                  const Tensor& masks) {
  const int8_t* masks_data = masks.data<int8_t>();
  int64_t num_mask = masks.dims()[0];
  int64_t height = masks.dims()[1];
  int64_t width = masks.dims()[2];
  int64_t num_pixel = width * height;

  Tensor boxes_from_masks;
  T* boxes_from_masks_data =
      boxes_from_masks.mutable_data<T>({num_mask, 4}, context.GetPlace());
  math::set_constant(context, &boxes_from_masks, 0);

  for (int64_t mask_id = 0; mask_id < num_mask; ++mask_id) {
    int xmin = width;
    int ymin = height;
    int xmax = 0;
    int ymax = 0;
    for (int64_t i = 0; i < num_pixel; ++i) {
      if (masks_data[mask_id * num_pixel + i] == 1) {
        int x = i % width;
        int y = i / width;
        xmin = x < xmin ? x : xmin;
        xmax = x > xmax ? x : xmax;
        ymin = y < ymin ? y : ymin;
        ymax = y > ymax ? y : ymax;
      }
    }
    boxes_from_masks_data[mask_id * kBoxDim] = static_cast<T>(xmin);
    boxes_from_masks_data[mask_id * kBoxDim + 1] = static_cast<T>(ymin);
    boxes_from_masks_data[mask_id * kBoxDim + 2] = static_cast<T>(xmax);
    boxes_from_masks_data[mask_id * kBoxDim + 3] = static_cast<T>(ymax);
  }

  return boxes_from_masks;
}

template <typename T>
static inline Tensor CropAndResize(const platform::CPUDeviceContext& context,
                                   const Tensor& masks_gt, const int mask_idx,
                                   const Tensor& rois_fg, const int roi_idx,
                                   const int M) {
  const int8_t* masks_gt_data = masks_gt.data<int8_t>();
  // int height = masks_gt.dims()[1];
  int width = masks_gt.dims()[2];
  const T* rois_fg_data = rois_fg.data<T>();

  Tensor result;
  int8_t* result_data = result.mutable_data<int8_t>({M, M}, context.GetPlace());

  T w = rois_fg_data[roi_idx + 2] - rois_fg_data[roi_idx + 0];
  T h = rois_fg_data[roi_idx + 3] - rois_fg_data[roi_idx + 1];
  w = std::max<T>(w, (T)1.);
  h = std::max<T>(h, (T)1.);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      int x = static_cast<int>(i / static_cast<T>(M) * w +
                               rois_fg_data[roi_idx + 0]);
      int y = static_cast<int>(j / static_cast<T>(M) * h +
                               rois_fg_data[roi_idx + 1]);
      result_data[j * M + i] = masks_gt_data[mask_idx + y * width + x] > 0;
    }
  }
  return result;
}

template <typename T>
static inline Tensor ExpandMaskTarget(const platform::CPUDeviceContext& context,
                                      const Tensor& masks,
                                      const Tensor& mask_class_labels,
                                      const int resolution,
                                      const int num_classes) {
  const int8_t* masks_data = masks.data<int8_t>();
  int64_t num_mask = masks.dims()[0];
  const int* mask_class_labels_data = mask_class_labels.data<int>();
  const int M = resolution * resolution;
  const int kMaskDim = M * num_classes;

  Tensor mask_targets;
  int* mask_targets_data =
      mask_targets.mutable_data<int>({num_mask, kMaskDim}, context.GetPlace());
  math::set_constant(context, &mask_targets, -1);
  for (int64_t mask_id = 0; mask_id < num_mask; ++mask_id) {
    int cls = mask_class_labels_data[mask_id];
    int start = M * cls;
    if (cls > 0) {
      for (int i = 0; i < M; ++i) {
        mask_targets_data[mask_id * kMaskDim + start + i] =
            static_cast<int>(masks_data[mask_id * M + i]);
      }
    }
  }
  return mask_targets;
}

template <typename T>
std::vector<Tensor> SampleMaskForOneImage(
    const platform::CPUDeviceContext& context, Tensor* im_info,
    Tensor* gt_classes, Tensor* is_crowd, Tensor* gt_segms, Tensor* rois,
    Tensor* label_int32, const int num_classes, const int resolution) {
  auto rois_et = framework::EigenTensor<T, 2>::From(*rois);
  auto im_scale = im_info->data<T>()[2];
  rois_et = rois_et / im_scale;
  // Prepare the mask targets by associating one gt mask to each training roi
  // that has a fg (non-bg) class label.
  std::vector<int> mask_gt_inds, fg_inds;
  const int64_t gt_size = static_cast<int64_t>(gt_segms->dims()[0]);
  const int64_t roi_size = static_cast<int64_t>(rois->dims()[0]);
  const int* gt_classes_data = gt_classes->data<int>();
  const int* is_crowd_data = is_crowd->data<int>();
  const int* label_int32_data = label_int32->data<int>();
  for (int64_t i = 0; i < gt_size; ++i) {
    if ((gt_classes_data[i] > 0) && (is_crowd_data[i] == 0)) {
      mask_gt_inds.emplace_back(i);
    }
  }
  for (int64_t i = 0; i < roi_size; ++i) {
    if (label_int32_data[i] > 0) {
      fg_inds.emplace_back(i);
    }
  }
  Tensor mask_gt_inds_t, fg_inds_t;
  int gt_num = mask_gt_inds.size();
  int fg_num = fg_inds.size();
  int* mask_gt_inds_data =
      mask_gt_inds_t.mutable_data<int>({gt_num}, context.GetPlace());
  int* fg_inds_data = fg_inds_t.mutable_data<int>({fg_num}, context.GetPlace());
  std::copy(mask_gt_inds.begin(), mask_gt_inds.end(), mask_gt_inds_data);
  std::copy(fg_inds.begin(), fg_inds.end(), fg_inds_data);
  Tensor masks_gt;
  masks_gt.mutable_data<int8_t>(
      {gt_num, gt_segms->dims()[1], gt_segms->dims()[2]}, context.GetPlace());
  CPUGather<int8_t>(context, *gt_segms, mask_gt_inds_t, &masks_gt);

  Tensor boxes_from_masks = MasksToBoxes<T>(context, masks_gt);
  std::vector<int> roi_has_mask =
      std::vector<int>(fg_inds.begin(), fg_inds.end());
  Tensor mask_class_labels;
  Tensor masks;
  Tensor rois_fg;

  if (fg_num > 0) {
    // Class labels for the foreground rois
    mask_class_labels.mutable_data<int>({fg_num, 1}, context.GetPlace());
    CPUGather<int>(context, *label_int32, fg_inds_t, &mask_class_labels);
    int8_t* masks_data = masks.mutable_data<int8_t>(
        {fg_num, resolution * resolution}, context.GetPlace());

    // Find overlap between all foreground rois and the bounding boxes
    // enclosing each segmentation
    rois_fg.mutable_data<T>({fg_num, 4}, context.GetPlace());
    CPUGather<T>(context, *rois, fg_inds_t, &rois_fg);
    Tensor overlaps_bbfg_bbmasks;
    overlaps_bbfg_bbmasks.mutable_data<T>({fg_num, gt_num}, context.GetPlace());
    BboxOverlaps<T>(rois_fg, boxes_from_masks, &overlaps_bbfg_bbmasks);

    // Map from each fg rois to the index of the mask with highest overlap
    // (measured by bbox overlap)
    T* overlaps_bbfg_bbmasks_data = overlaps_bbfg_bbmasks.data<T>();
    std::vector<int> fg_masks_inds;
    float epsilon = 0.00001;
    for (int64_t i = 0; i < fg_num; ++i) {
      const T* v = overlaps_bbfg_bbmasks_data + i * gt_num;
      T max_overlap = *std::max_element(v, v + gt_num);
      for (int64_t j = 0; j < gt_num; ++j) {
        T val = overlaps_bbfg_bbmasks_data[i * gt_num + j];
        auto diff = std::abs(max_overlap - val);
        if (diff < epsilon) {
          fg_masks_inds.emplace_back(j);
          break;
        }
      }
    }

    // add fg targets
    for (int64_t i = 0; i < fg_num; ++i) {
      int fg_masks_ind = fg_masks_inds[i];
      int mask_idx = fg_masks_ind * gt_segms->dims()[1] * gt_segms->dims()[2];
      int roi_idx = i * kBoxDim;
      Tensor mask = CropAndResize<T>(context, masks_gt, mask_idx, rois_fg,
                                     roi_idx, resolution);
      int8_t* mask_data = mask.data<int8_t>();
      int offset = i * resolution * resolution;
      memcpy(masks_data + offset, mask_data, mask.numel() * sizeof(int8_t));
    }
  } else {
    // The network cannot handle empty blobs, so we must provide a mask
    // We simply take the first bg roi, given it an all -1's mask (ignore
    // label), and label it with class zero (bg).
    std::vector<int> bg_inds;
    for (int64_t i = 0; i < roi_size; ++i) {
      if (label_int32_data[i] == 0) {
        bg_inds.emplace_back(i);
        break;
      }
    }
    int bg_num = 1;
    bg_inds = std::vector<int>(bg_inds.begin(), bg_inds.begin() + bg_num);
    Tensor bg_inds_t;
    int* bg_inds_data =
        bg_inds_t.mutable_data<int>({bg_num}, context.GetPlace());
    std::copy(bg_inds.begin(), bg_inds.end(), bg_inds_data);
    rois_fg.mutable_data<T>({bg_num, 4}, context.GetPlace());
    CPUGather<T>(context, *rois, bg_inds_t, &rois_fg);
    masks.mutable_data<int>({bg_num, resolution * resolution},
                            context.GetPlace());
    math::set_constant(context, &masks, -1);
    int* mask_class_labels_data =
        mask_class_labels.mutable_data<int>({bg_num, 1}, context.GetPlace());
    mask_class_labels_data[0] = 0;
    roi_has_mask = std::vector<int>(bg_inds.begin(), bg_inds.end());
  }

  Tensor masks_expand = ExpandMaskTarget<T>(context, masks, mask_class_labels,
                                            resolution, num_classes);
  auto rois_fg_et = framework::EigenTensor<T, 2>::From(rois_fg);
  rois_fg_et = rois_fg_et * im_scale;
  Tensor roi_has_mask_t;
  int roi_has_mask_size = roi_has_mask.size();
  int* roi_has_mask_data = roi_has_mask_t.mutable_data<int>(
      {roi_has_mask_size, 1}, context.GetPlace());
  std::copy(roi_has_mask.begin(), roi_has_mask.end(), roi_has_mask_data);
  std::vector<Tensor> res;
  res.emplace_back(rois_fg);
  res.emplace_back(roi_has_mask_t);
  res.emplace_back(masks_expand);
  return res;
}

template <typename T>
class GenerateMaskLabelsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* im_info = context.Input<LoDTensor>("ImInfo");
    auto* gt_classes = context.Input<LoDTensor>("GtClasses");
    auto* is_crowd = context.Input<LoDTensor>("IsCrowd");
    auto* gt_segms = context.Input<LoDTensor>("GtSegms");
    auto* rois = context.Input<LoDTensor>("Rois");
    auto* label_int32 = context.Input<LoDTensor>("LabelsInt32");

    auto* mask_rois = context.Output<LoDTensor>("MaskRois");
    auto* roi_has_mask_int32 = context.Output<LoDTensor>("RoiHasMaskInt32");
    auto* mask_int32 = context.Output<LoDTensor>("MaskInt32");

    int num_classes = context.Attr<int>("num_classes");
    int resolution = context.Attr<int>("resolution");

    PADDLE_ENFORCE_EQ(gt_classes->lod().size(), 1UL,
                      "GenerateMaskLabelsOp gt_classes needs 1 level of LoD");
    PADDLE_ENFORCE_EQ(is_crowd->lod().size(), 1UL,
                      "GenerateMaskLabelsOp is_crowd needs 1 level of LoD");
    PADDLE_ENFORCE_EQ(gt_segms->lod().size(), 1UL,
                      "GenerateMaskLabelsOp gt_segms needs 1 level of LoD");
    PADDLE_ENFORCE_EQ(rois->lod().size(), 1UL,
                      "GenerateMaskLabelsOp rois needs 1 level of LoD");
    PADDLE_ENFORCE_EQ(label_int32->lod().size(), 1UL,
                      "GenerateMaskLabelsOp label_int32 needs 1 level of LoD");

    int64_t n = static_cast<int64_t>(gt_segms->lod().back().size() - 1);
    int kMaskDim = num_classes * resolution * resolution;

    mask_rois->mutable_data<T>({rois->numel(), kBoxDim}, context.GetPlace());
    roi_has_mask_int32->mutable_data<int>({rois->numel(), 1},
                                          context.GetPlace());
    mask_int32->mutable_data<int>({rois->numel(), kMaskDim},
                                  context.GetPlace());

    framework::LoD lod;
    std::vector<size_t> lod0(1, 0);

    int64_t num_mask = 0;
    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();

    auto gt_classes_lod = gt_classes->lod().back();
    auto is_crowd_lod = is_crowd->lod().back();
    auto gt_segms_lod = gt_segms->lod().back();
    auto rois_lod = rois->lod().back();
    auto label_int32_lod = label_int32->lod().back();

    for (int i = 0; i < n; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor gt_classes_slice =
          gt_classes->Slice(gt_classes_lod[i], gt_classes_lod[i + 1]);
      Tensor is_crowd_slice =
          is_crowd->Slice(is_crowd_lod[i], is_crowd_lod[i + 1]);
      Tensor gt_segms_slice =
          gt_segms->Slice(gt_segms_lod[i], gt_segms_lod[i + 1]);
      Tensor label_int32_slice =
          label_int32->Slice(label_int32_lod[i], label_int32_lod[i + 1]);
      Tensor rois_slice = rois->Slice(rois_lod[i], rois_lod[i + 1]);
      std::vector<Tensor> tensor_output = SampleMaskForOneImage<T>(
          dev_ctx, &im_info_slice, &gt_classes_slice, &is_crowd_slice,
          &gt_segms_slice, &rois_slice, &label_int32_slice, num_classes,
          resolution);
      Tensor sampled_mask_rois = tensor_output[0];
      Tensor sampled_roi_has_mask_int32 = tensor_output[1];
      Tensor sampled_mask_int32 = tensor_output[2];

      AppendMask<T>(mask_rois, kBoxDim * num_mask, &sampled_mask_rois);
      AppendMask<int>(roi_has_mask_int32, num_mask,
                      &sampled_roi_has_mask_int32);
      AppendMask<int>(mask_int32, kMaskDim * num_mask, &sampled_mask_int32);

      num_mask += sampled_mask_rois.dims()[0];
      lod0.emplace_back(num_mask);
    }

    lod.emplace_back(lod0);
    mask_rois->set_lod(lod);
    roi_has_mask_int32->set_lod(lod);
    mask_int32->set_lod(lod);
    mask_rois->Resize({num_mask, kBoxDim});
    roi_has_mask_int32->Resize({num_mask, 1});
    mask_int32->Resize({num_mask, kMaskDim});
  }
};

class GenerateMaskLabelsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("ImInfo",
             "(Tensor), This input is a 2D Tensor with shape [B, 3]. "
             "B is the number of input images, "
             "each element consists of im_height, im_width, im_scale.");
    AddInput("GtClasses",
             "(LoDTensor), This input is a 2D LoDTensor with shape [M, 1]. "
             "M is the number of groundtruth, "
             "each element is a class label of groundtruth.");
    AddInput(
        "IsCrowd",
        "(LoDTensor), This input is a 2D LoDTensor with shape [M, 1]. "
        "M is the number of groundtruth, "
        "each element is a flag indicates whether a groundtruth is crowd.");
    AddInput("GtSegms",
             "(LoDTensor), This input is a 4D LoDTensor with shape [M, H, W]. "
             "M is the number of groundtruth, "
             "H and W is height and width of image, respectively.");
    AddInput(
        "Rois",
        "(LoDTensor), This input is a 2D LoDTensor with shape [R, 4]. "
        "R is the number of rois which is the output of "
        "generate_proposal_labels, "
        "each element is a bounding box with [xmin, ymin, xmax, ymax] format.");
    AddInput("LabelsInt32",
             "(LoDTensor), This intput is a 2D LoDTensor with shape [R, 1], "
             "each element repersents a class label of a roi");

    AddOutput(
        "MaskRois",
        "(LoDTensor), This output is a 2D LoDTensor with shape [P, 4]. "
        "P is the number of mask, "
        "each element is a bounding box with [xmin, ymin, xmax, ymax] format.");
    AddOutput("RoiHasMaskInt32",
              "(LoDTensor), This output is a 2D LoDTensor with shape [P, 1], "
              "each element repersents the output mask rois index with regard "
              "to input rois");
    AddOutput("MaskInt32",
              "(LoDTensor), This output is a 4D LoDTensor with shape [P, Q], "
              "Q equal to num_classes * resolution * resolution");

    AddAttr<int>("num_classes", "Class number.");
    AddAttr<int>("resolution", "Resolution of mask.");

    AddComment(R"DOC(
GenerateMaskLabelsOp
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(generate_mask_labels, ops::GenerateMaskLabelsOp,
                  ops::GenerateMaskLabelsOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(generate_mask_labels,
                       ops::GenerateMaskLabelsKernel<float>,
                       ops::GenerateMaskLabelsKernel<double>);
