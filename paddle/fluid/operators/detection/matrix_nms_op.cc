/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/detection/nms_util.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class MatrixNMSOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("BBoxes"), "Input", "BBoxes", "MatrixNMS");
    OP_INOUT_CHECK(ctx->HasInput("Scores"), "Input", "Scores", "MatrixNMS");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "MatrixNMS");
    auto box_dims = ctx->GetInputDim("BBoxes");
    auto score_dims = ctx->GetInputDim("Scores");
    auto score_size = score_dims.size();

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(score_size == 3,
                        true,
                        platform::errors::InvalidArgument(
                            "The rank of Input(Scores) must be 3. "
                            "But received rank = %d.",
                            score_size));
      PADDLE_ENFORCE_EQ(box_dims.size(),
                        3,
                        platform::errors::InvalidArgument(
                            "The rank of Input(BBoxes) must be 3."
                            "But received rank = %d.",
                            box_dims.size()));
      PADDLE_ENFORCE_EQ(box_dims[2] == 4,
                        true,
                        platform::errors::InvalidArgument(
                            "The last dimension of Input (BBoxes) must be 4, "
                            "represents the layout of coordinate "
                            "[xmin, ymin, xmax, ymax]."));
      PADDLE_ENFORCE_EQ(
          box_dims[1],
          score_dims[2],
          platform::errors::InvalidArgument(
              "The 2nd dimension of Input(BBoxes) must be equal to "
              "last dimension of Input(Scores), which represents the "
              "predicted bboxes."
              "But received box_dims[1](%s) != socre_dims[2](%s)",
              box_dims[1],
              score_dims[2]));
    }
    ctx->SetOutputDim("Out", {box_dims[1], box_dims[2] + 2});
    ctx->SetOutputDim("Index", {box_dims[1], 1});
    if (ctx->HasOutput("RoisNum")) {
      ctx->SetOutputDim("RoisNum", {-1});
    }
    if (!ctx->IsRuntime()) {
      ctx->SetLoDLevel("Out", std::max(ctx->GetLoDLevel("BBoxes"), 1));
      ctx->SetLoDLevel("Index", std::max(ctx->GetLoDLevel("BBoxes"), 1));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Scores"),
        platform::CPUPlace());
  }
};

template <typename T, bool gaussian>
struct decay_score;

template <typename T>
struct decay_score<T, true> {
  T operator()(T iou, T max_iou, T sigma) {
    return std::exp((max_iou * max_iou - iou * iou) * sigma);
  }
};

template <typename T>
struct decay_score<T, false> {
  T operator()(T iou, T max_iou, T sigma) {
    return (1. - iou) / (1. - max_iou);
  }
};

template <typename T, bool gaussian>
void NMSMatrix(const Tensor& bbox,
               const Tensor& scores,
               const T score_threshold,
               const T post_threshold,
               const float sigma,
               const int64_t top_k,
               const bool normalized,
               std::vector<int>* selected_indices,
               std::vector<T>* decayed_scores) {
  int64_t num_boxes = bbox.dims()[0];
  int64_t box_size = bbox.dims()[1];

  auto score_ptr = scores.data<T>();
  auto bbox_ptr = bbox.data<T>();

  std::vector<int32_t> perm(num_boxes);
  std::iota(perm.begin(), perm.end(), 0);
  auto end = std::remove_if(
      perm.begin(), perm.end(), [&score_ptr, score_threshold](int32_t idx) {
        return score_ptr[idx] <= score_threshold;
      });

  auto sort_fn = [&score_ptr](int32_t lhs, int32_t rhs) {
    return score_ptr[lhs] > score_ptr[rhs];
  };

  int64_t num_pre = std::distance(perm.begin(), end);
  if (num_pre <= 0) {
    return;
  }
  if (top_k > -1 && num_pre > top_k) {
    num_pre = top_k;
  }
  std::partial_sort(perm.begin(), perm.begin() + num_pre, end, sort_fn);

  std::vector<T> iou_matrix((num_pre * (num_pre - 1)) >> 1);
  std::vector<T> iou_max(num_pre);

  iou_max[0] = 0.;
  for (int64_t i = 1; i < num_pre; i++) {
    T max_iou = 0.;
    auto idx_a = perm[i];
    for (int64_t j = 0; j < i; j++) {
      auto idx_b = perm[j];
      auto iou = JaccardOverlap<T>(
          bbox_ptr + idx_a * box_size, bbox_ptr + idx_b * box_size, normalized);
      max_iou = std::max(max_iou, iou);
      iou_matrix[i * (i - 1) / 2 + j] = iou;
    }
    iou_max[i] = max_iou;
  }

  if (score_ptr[perm[0]] > post_threshold) {
    selected_indices->push_back(perm[0]);
    decayed_scores->push_back(score_ptr[perm[0]]);
  }

  decay_score<T, gaussian> decay_fn;
  for (int64_t i = 1; i < num_pre; i++) {
    T min_decay = 1.;
    for (int64_t j = 0; j < i; j++) {
      auto max_iou = iou_max[j];
      auto iou = iou_matrix[i * (i - 1) / 2 + j];
      auto decay = decay_fn(iou, max_iou, sigma);
      min_decay = std::min(min_decay, decay);
    }
    auto ds = min_decay * score_ptr[perm[i]];
    if (ds <= post_threshold) continue;
    selected_indices->push_back(perm[i]);
    decayed_scores->push_back(ds);
  }
}

template <typename T>
class MatrixNMSKernel : public framework::OpKernel<T> {
 public:
  size_t MultiClassMatrixNMS(const Tensor& scores,
                             const Tensor& bboxes,
                             std::vector<T>* out,
                             std::vector<int>* indices,
                             int start,
                             int64_t background_label,
                             int64_t nms_top_k,
                             int64_t keep_top_k,
                             bool normalized,
                             T score_threshold,
                             T post_threshold,
                             bool use_gaussian,
                             float gaussian_sigma) const {
    std::vector<int> all_indices;
    std::vector<T> all_scores;
    std::vector<T> all_classes;
    all_indices.reserve(scores.numel());
    all_scores.reserve(scores.numel());
    all_classes.reserve(scores.numel());

    size_t num_det = 0;
    auto class_num = scores.dims()[0];
    Tensor score_slice;
    for (int64_t c = 0; c < class_num; ++c) {
      if (c == background_label) continue;
      score_slice = scores.Slice(c, c + 1);
      if (use_gaussian) {
        NMSMatrix<T, true>(bboxes,
                           score_slice,
                           score_threshold,
                           post_threshold,
                           gaussian_sigma,
                           nms_top_k,
                           normalized,
                           &all_indices,
                           &all_scores);
      } else {
        NMSMatrix<T, false>(bboxes,
                            score_slice,
                            score_threshold,
                            post_threshold,
                            gaussian_sigma,
                            nms_top_k,
                            normalized,
                            &all_indices,
                            &all_scores);
      }
      for (size_t i = 0; i < all_indices.size() - num_det; i++) {
        all_classes.push_back(static_cast<T>(c));
      }
      num_det = all_indices.size();
    }

    if (num_det <= 0) {
      return num_det;
    }

    if (keep_top_k > -1) {
      auto k = static_cast<size_t>(keep_top_k);
      if (num_det > k) num_det = k;
    }

    std::vector<int32_t> perm(all_indices.size());
    std::iota(perm.begin(), perm.end(), 0);

    std::partial_sort(perm.begin(),
                      perm.begin() + num_det,
                      perm.end(),
                      [&all_scores](int lhs, int rhs) {
                        return all_scores[lhs] > all_scores[rhs];
                      });

    for (size_t i = 0; i < num_det; i++) {
      auto p = perm[i];
      auto idx = all_indices[p];
      auto cls = all_classes[p];
      auto score = all_scores[p];
      auto bbox = bboxes.data<T>() + idx * bboxes.dims()[1];
      (*indices).push_back(start + idx);
      (*out).push_back(cls);
      (*out).push_back(score);
      for (int j = 0; j < bboxes.dims()[1]; j++) {
        (*out).push_back(bbox[j]);
      }
    }

    return num_det;
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* boxes = ctx.Input<LoDTensor>("BBoxes");
    auto* scores = ctx.Input<LoDTensor>("Scores");
    auto* outs = ctx.Output<LoDTensor>("Out");
    auto* index = ctx.Output<LoDTensor>("Index");

    auto background_label = ctx.Attr<int>("background_label");
    auto nms_top_k = ctx.Attr<int>("nms_top_k");
    auto keep_top_k = ctx.Attr<int>("keep_top_k");
    auto normalized = ctx.Attr<bool>("normalized");
    auto score_threshold = ctx.Attr<float>("score_threshold");
    auto post_threshold = ctx.Attr<float>("post_threshold");
    auto use_gaussian = ctx.Attr<bool>("use_gaussian");
    auto gaussian_sigma = ctx.Attr<float>("gaussian_sigma");

    auto score_dims = scores->dims();
    auto batch_size = score_dims[0];
    auto num_boxes = score_dims[2];
    auto box_dim = boxes->dims()[2];
    auto out_dim = box_dim + 2;

    Tensor boxes_slice, scores_slice;
    size_t num_out = 0;
    std::vector<size_t> offsets = {0};
    std::vector<T> detections;
    std::vector<int> indices;
    std::vector<int> num_per_batch;
    detections.reserve(out_dim * num_boxes * batch_size);
    indices.reserve(num_boxes * batch_size);
    num_per_batch.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      scores_slice = scores->Slice(i, i + 1);
      scores_slice.Resize({score_dims[1], score_dims[2]});
      boxes_slice = boxes->Slice(i, i + 1);
      boxes_slice.Resize({score_dims[2], box_dim});
      int start = i * score_dims[2];
      num_out = MultiClassMatrixNMS(scores_slice,
                                    boxes_slice,
                                    &detections,
                                    &indices,
                                    start,
                                    background_label,
                                    nms_top_k,
                                    keep_top_k,
                                    normalized,
                                    score_threshold,
                                    post_threshold,
                                    use_gaussian,
                                    gaussian_sigma);
      offsets.push_back(offsets.back() + num_out);
      num_per_batch.emplace_back(num_out);
    }

    int64_t num_kept = offsets.back();
    if (num_kept == 0) {
      outs->mutable_data<T>({0, out_dim}, ctx.GetPlace());
      index->mutable_data<int>({0, 1}, ctx.GetPlace());
    } else {
      outs->mutable_data<T>({num_kept, out_dim}, ctx.GetPlace());
      index->mutable_data<int>({num_kept, 1}, ctx.GetPlace());
      std::copy(detections.begin(), detections.end(), outs->data<T>());
      std::copy(indices.begin(), indices.end(), index->data<int>());
    }

    if (ctx.HasOutput("RoisNum")) {
      auto* rois_num = ctx.Output<Tensor>("RoisNum");
      rois_num->mutable_data<int>({batch_size}, ctx.GetPlace());
      std::copy(
          num_per_batch.begin(), num_per_batch.end(), rois_num->data<int>());
    }
    framework::LoD lod;
    lod.emplace_back(offsets);
    outs->set_lod(lod);
    index->set_lod(lod);
  }
};

class MatrixNMSOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("BBoxes",
             "(Tensor) A 3-D Tensor with shape "
             "[N, M, 4] represents the predicted locations of M bounding boxes"
             ", N is the batch size. "
             "Each bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax], when box size equals to 4.");
    AddInput("Scores",
             "(Tensor) A 3-D Tensor with shape [N, C, M] represents the "
             "predicted confidence predictions. N is the batch size, C is the "
             "class number, M is number of bounding boxes. For each category "
             "there are total M scores which corresponding M bounding boxes. "
             " Please note, M is equal to the 2nd dimension of BBoxes. ");
    AddAttr<int>(
        "background_label",
        "(int, default: 0) "
        "The index of background label, the background label will be ignored. "
        "If set to -1, then all categories will be considered.")
        .SetDefault(0);
    AddAttr<float>("score_threshold",
                   "(float) "
                   "Threshold to filter out bounding boxes with low "
                   "confidence score.");
    AddAttr<float>("post_threshold",
                   "(float, default 0.) "
                   "Threshold to filter out bounding boxes with low "
                   "confidence score AFTER decaying.")
        .SetDefault(0.);
    AddAttr<int>("nms_top_k",
                 "(int64_t) "
                 "Maximum number of detections to be kept according to the "
                 "confidences after the filtering detections based on "
                 "score_threshold");
    AddAttr<int>("keep_top_k",
                 "(int64_t) "
                 "Number of total bboxes to be kept per image after NMS "
                 "step. -1 means keeping all bboxes after NMS step.");
    AddAttr<bool>("normalized",
                  "(bool, default true) "
                  "Whether detections are normalized.")
        .SetDefault(true);
    AddAttr<bool>("use_gaussian",
                  "(bool, default false) "
                  "Whether to use Gaussian as decreasing function.")
        .SetDefault(false);
    AddAttr<float>("gaussian_sigma",
                   "(float) "
                   "Sigma for Gaussian decreasing function, only takes effect ",
                   "when 'use_gaussian' is enabled.")
        .SetDefault(2.);
    AddOutput("Out",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 6] represents the "
              "detections. Each row has 6 values: "
              "[label, confidence, xmin, ymin, xmax, ymax]. "
              "the offsets in first dimension are called LoD, the number of "
              "offset is N + 1, if LoD[i + 1] - LoD[i] == 0, means there is "
              "no detected bbox.");
    AddOutput("Index",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 1] represents the "
              "index of selected bbox. The index is the absolute index cross "
              "batches.");
    AddOutput("RoisNum", "(Tensor), Number of RoIs in each images.")
        .AsDispensable();
    AddComment(R"DOC(
This operator does multi-class matrix non maximum suppression (NMS) on batched
boxes and scores.
In the NMS step, this operator greedily selects a subset of detection bounding
boxes that have high scores larger than score_threshold, if providing this
threshold, then selects the largest nms_top_k confidences scores if nms_top_k
is larger than -1. Then this operator decays boxes score according to the
Matrix NMS scheme.
Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
per image if keep_top_k is larger than -1.
This operator support multi-class and batched inputs. It applying NMS
independently for each class. The outputs is a 2-D LoDTenosr, for each
image, the offsets in first dimension of LoDTensor are called LoD, the number
of offset is N + 1, where N is the batch size. If LoD[i + 1] - LoD[i] == 0,
means there is no detected bbox for this image. Now this operator has one more
output, which is RoisNum. The size of RoisNum is N, RoisNum[i] means the number of 
detected bbox for this image.

For more information on Matrix NMS, please refer to:
https://arxiv.org/abs/2003.10152
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    matrix_nms,
    ops::MatrixNMSOp,
    ops::MatrixNMSOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(matrix_nms,
                       ops::MatrixNMSKernel<float>,
                       ops::MatrixNMSKernel<double>);
REGISTER_OP_VERSION(matrix_nms)
    .AddCheckpoint(R"ROC(Upgrade matrix_nms: add a new output [RoisNum].)ROC",
                   paddle::framework::compatible::OpVersionDesc().NewOutput(
                       "RoisNum", "The number of RoIs in each image."));
