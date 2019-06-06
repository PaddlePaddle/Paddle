/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
limitations under the License. */

#include <glog/logging.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/poly_util.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class RetinanetDetectionOutputOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("BBoxes"),
        "Input(BBoxes) of RetinanetDetectionOutput should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Scores"),
        "Input(Scores) of RetinanetDetectionOutput should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Anchors"),
        "Input(Anchors) of RetinanetDetectionOutput should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("ImInfo"),
        "Input(ImInfo) of RetinanetDetectionOutput should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out) of RetinanetDetectionOutput should not be null.");

    auto box_dims = ctx->GetInputDim("BBoxes");
    auto score_dims = ctx->GetInputDim("Scores");
    auto anchor_dims = ctx->GetInputDim("Anchors");
    auto im_info_dims = ctx->GetInputDim("ImInfo");

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(score_dims.size(), 3,
                        "The rank of Input(Scores) must be 3");
      PADDLE_ENFORCE_EQ(box_dims.size(), 3,
                        "The rank of Input(BBoxes) must be 3");
      PADDLE_ENFORCE_EQ(anchor_dims.size(), 2,
                        "The rank of Input(Anchors) must be 2");
      PADDLE_ENFORCE(box_dims[2] == 4,
                     "The last dimension of Input(BBoxes) must be 4, "
                     "represents the layout of coordinate "
                     "[xmin, ymin, xmax, ymax]");
      PADDLE_ENFORCE_EQ(box_dims[1], score_dims[1],
                        "The 2nd dimension of Input(BBoxes) must be equal to "
                        "2nd dimension of Input(Scores), which represents the "
                        "number of the predicted boxes.");

      PADDLE_ENFORCE_EQ(anchor_dims[0], box_dims[1],
                        "The 1st dimension of Input(Anchors) must be equal to "
                        "2nd dimension of Input(BBoxes), which represents the "
                        "number of the predicted boxes.");
      PADDLE_ENFORCE_EQ(im_info_dims.size(), 2,
                        "The rank of Input(ImInfo) must be 2.");
    }
    // Here the box_dims[0] is not the real dimension of output.
    // It will be rewritten in the computing kernel.
    ctx->SetOutputDim("Out", {box_dims[1], box_dims[2] + 2});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::Tensor>("Scores")->type(), platform::CPUPlace());
  }
};

template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
bool SortScoreTwoPairDescend(const std::pair<float, std::pair<T, T>>& pair1,
                             const std::pair<float, std::pair<T, T>>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
static inline void GetMaxScoreIndex(
    const std::vector<T>& scores, const T threshold, int top_k,
    std::vector<std::pair<T, int>>* sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

template <class T>
static inline T BBoxArea(const std::vector<T>& box, const bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

template <class T>
static inline T JaccardOverlap(const std::vector<T>& box1,
                               const std::vector<T>& box2,
                               const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    T norm = normalized ? static_cast<T>(0.) : static_cast<T>(1.);
    T inter_w = inter_xmax - inter_xmin + norm;
    T inter_h = inter_ymax - inter_ymin + norm;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <typename T>
class RetinanetDetectionOutputKernel : public framework::OpKernel<T> {
 public:
  void NMSFast(const std::vector<std::vector<T>>& cls_dets,
               const T nms_threshold, const T eta,
               std::vector<int>* selected_indices) const {
    int64_t num_boxes = cls_dets.size();
    std::vector<std::pair<T, int>> sorted_indices;
    for (int64_t i = 0; i < num_boxes; ++i) {
      sorted_indices.push_back(std::make_pair(cls_dets[i][4], i));
    }
    // Sort the score pair according to the scores in descending order
    std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
                     SortScorePairDescend<int>);
    selected_indices->clear();
    T adaptive_threshold = nms_threshold;

    while (sorted_indices.size() != 0) {
      const int idx = sorted_indices.front().second;
      bool keep = true;
      for (size_t k = 0; k < selected_indices->size(); ++k) {
        if (keep) {
          const int kept_idx = (*selected_indices)[k];
          T overlap = T(0.);

          overlap = JaccardOverlap<T>(cls_dets[idx], cls_dets[kept_idx], false);
          keep = overlap <= adaptive_threshold;
        } else {
          break;
        }
      }
      if (keep) {
        selected_indices->push_back(idx);
      }
      sorted_indices.erase(sorted_indices.begin());
      if (keep && eta < 1 && adaptive_threshold > 0.5) {
        adaptive_threshold *= eta;
      }
    }
  }

  void DeltaScoreToPrediction(
      const std::vector<T>& bboxes_data, const std::vector<T>& anchors_data,
      T im_height, T im_width, T im_scale, int class_num,
      const std::vector<std::pair<T, int>>& sorted_indices,
      std::map<int, std::vector<std::vector<T>>>* preds) const {
    im_height = static_cast<T>(round(im_height / im_scale));
    im_width = static_cast<T>(round(im_width / im_scale));
    T zero(0);
    int i = 0;
    for (const auto& it : sorted_indices) {
      T score = it.first;
      int idx = it.second;
      int a = idx / class_num;
      int c = idx % class_num;

      int box_offset = a * 4;
      T anchor_box_width =
          anchors_data[box_offset + 2] - anchors_data[box_offset] + 1;
      T anchor_box_height =
          anchors_data[box_offset + 3] - anchors_data[box_offset + 1] + 1;
      T anchor_box_center_x = anchors_data[box_offset] + anchor_box_width / 2;
      T anchor_box_center_y =
          anchors_data[box_offset + 1] + anchor_box_height / 2;
      T target_box_center_x = 0, target_box_center_y = 0;
      T target_box_width = 0, target_box_height = 0;
      target_box_center_x =
          bboxes_data[box_offset] * anchor_box_width + anchor_box_center_x;
      target_box_center_y =
          bboxes_data[box_offset + 1] * anchor_box_height + anchor_box_center_y;
      target_box_width =
          std::exp(bboxes_data[box_offset + 2]) * anchor_box_width;
      target_box_height =
          std::exp(bboxes_data[box_offset + 3]) * anchor_box_height;
      T pred_box_xmin = target_box_center_x - target_box_width / 2;
      T pred_box_ymin = target_box_center_y - target_box_height / 2;
      T pred_box_xmax = target_box_center_x + target_box_width / 2 - 1;
      T pred_box_ymax = target_box_center_y + target_box_height / 2 - 1;
      pred_box_xmin = pred_box_xmin / im_scale;
      pred_box_ymin = pred_box_ymin / im_scale;
      pred_box_xmax = pred_box_xmax / im_scale;
      pred_box_ymax = pred_box_ymax / im_scale;

      pred_box_xmin = std::max(std::min(pred_box_xmin, im_width - 1), zero);
      pred_box_ymin = std::max(std::min(pred_box_ymin, im_height - 1), zero);
      pred_box_xmax = std::max(std::min(pred_box_xmax, im_width - 1), zero);
      pred_box_ymax = std::max(std::min(pred_box_ymax, im_height - 1), zero);

      std::vector<T> one_pred;
      one_pred.push_back(pred_box_xmin);
      one_pred.push_back(pred_box_ymin);
      one_pred.push_back(pred_box_xmax);
      one_pred.push_back(pred_box_ymax);
      one_pred.push_back(score);
      (*preds)[c].push_back(one_pred);
      i++;
    }
  }

  void MultiClassNMS(const std::map<int, std::vector<std::vector<T>>>& preds,
                     int class_num, const int keep_top_k, const T nms_threshold,
                     const T nms_eta, std::vector<std::vector<T>>* nmsed_out,
                     int* num_nmsed_out) const {
    std::map<int, std::vector<int>> indices;
    int num_det = 0;
    for (int c = 0; c < class_num; ++c) {
      if (static_cast<bool>(preds.count(c))) {
        const std::vector<std::vector<T>> cls_dets = preds.at(c);
        NMSFast(cls_dets, nms_threshold, nms_eta, &(indices[c]));
        num_det += indices[c].size();
      }
    }

    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
    for (const auto& it : indices) {
      int label = it.first;
      const std::vector<int>& label_indices = it.second;
      for (size_t j = 0; j < label_indices.size(); ++j) {
        int idx = label_indices[j];
        score_index_pairs.push_back(std::make_pair(preds.at(label)[idx][4],
                                                   std::make_pair(label, idx)));
      }
    }
    // Keep top k results per image.
    std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                     SortScoreTwoPairDescend<int>);
    if (num_det > keep_top_k) {
      score_index_pairs.resize(keep_top_k);
    }

    // Store the new indices.
    std::map<int, std::vector<int>> new_indices;
    for (const auto& it : score_index_pairs) {
      int label = it.second.first;
      int idx = it.second.second;
      std::vector<T> one_pred;
      one_pred.push_back(label);
      one_pred.push_back(preds.at(label)[idx][4]);
      one_pred.push_back(preds.at(label)[idx][0]);
      one_pred.push_back(preds.at(label)[idx][1]);
      one_pred.push_back(preds.at(label)[idx][2]);
      one_pred.push_back(preds.at(label)[idx][3]);
      nmsed_out->push_back(one_pred);
    }

    *num_nmsed_out = (num_det > keep_top_k ? keep_top_k : num_det);
  }

  void RetinanetDetectionOutput(const framework::ExecutionContext& ctx,
                                const Tensor& scores, const Tensor& bboxes,
                                const Tensor* anchors, const Tensor& im_info,
                                std::vector<std::vector<T>>* nmsed_out,
                                int* num_nmsed_out) const {
    int64_t min_level = ctx.Attr<int>("min_level");
    int64_t max_level = ctx.Attr<int>("max_level");
    int64_t nms_top_k = ctx.Attr<int>("nms_top_k");
    int64_t keep_top_k = ctx.Attr<int>("keep_top_k");
    T nms_threshold = static_cast<T>(ctx.Attr<float>("nms_threshold"));
    T nms_eta = static_cast<T>(ctx.Attr<float>("nms_eta"));
    T score_threshold = static_cast<T>(ctx.Attr<float>("score_threshold"));

    int64_t class_num = scores.dims()[1];
    // The number of total boxes from all FPN level
    int64_t total_cell_num = bboxes.numel() / bboxes.dims()[1];
    int64_t factors = 0;
    for (int64_t l = min_level; l < (max_level + 1); ++l) {
      factors += std::pow(2, max_level - l) * std::pow(2, max_level - l);
    }
    // The number of boxes from the highest FPN level
    int64_t coarsest_cell_num = total_cell_num / factors;
    std::map<int, std::vector<std::vector<T>>> preds;
    int begin_idx = 0;
    int end_idx = 0;
    for (int64_t l = min_level; l < (max_level + 1); ++l) {
      int factor = std::pow(2, max_level - l);
      // The box number of the l-th level is 4**(max_level - l) times of
      // that of the highest FPN level
      begin_idx = end_idx;
      end_idx = begin_idx + coarsest_cell_num * factor * factor;
      // Fetch per level score
      Tensor scores_per_level = scores.Slice(begin_idx, end_idx);
      // Fetch per level bbox
      Tensor bboxes_per_level = bboxes.Slice(begin_idx, end_idx);
      // Fetch per level anchor
      Tensor anchors_per_level = anchors->Slice(begin_idx, end_idx);

      int64_t scores_num = scores_per_level.numel();
      int64_t bboxes_num = bboxes_per_level.numel();
      std::vector<T> scores_data(scores_num);
      std::vector<T> bboxes_data(bboxes_num);
      std::vector<T> anchors_data(bboxes_num);
      std::copy_n(scores_per_level.data<T>(), scores_num, scores_data.begin());
      std::copy_n(bboxes_per_level.data<T>(), bboxes_num, bboxes_data.begin());
      std::copy_n(anchors_per_level.data<T>(), bboxes_num,
                  anchors_data.begin());
      std::vector<std::pair<T, int>> sorted_indices;

      // For the highest level, we take the threshold 0.0
      T threshold = (l < max_level ? score_threshold : 0.0);
      GetMaxScoreIndex(scores_data, threshold, nms_top_k, &sorted_indices);
      auto* im_info_data = im_info.data<T>();
      auto im_height = im_info_data[0];
      auto im_width = im_info_data[1];
      auto im_scale = im_info_data[2];
      DeltaScoreToPrediction(bboxes_data, anchors_data, im_height, im_width,
                             im_scale, class_num, sorted_indices, &preds);
    }

    MultiClassNMS(preds, class_num, keep_top_k, nms_threshold, nms_eta,
                  nmsed_out, num_nmsed_out);
  }

  void MultiClassOutput(const platform::DeviceContext& ctx,
                        const std::vector<std::vector<T>>& nmsed_out,
                        Tensor* outs) const {
    auto* odata = outs->data<T>();
    int count = 0;
    int64_t out_dim = 6;
    for (size_t i = 0; i < nmsed_out.size(); ++i) {
      odata[count * out_dim] = nmsed_out[i][0];      // label
      odata[count * out_dim + 1] = nmsed_out[i][1];  // score
      odata[count * out_dim + 2] = nmsed_out[i][2];  // xmin
      odata[count * out_dim + 3] = nmsed_out[i][3];  // xmin
      odata[count * out_dim + 4] = nmsed_out[i][4];  // xmin
      odata[count * out_dim + 5] = nmsed_out[i][5];  // xmin
      count++;
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* boxes = ctx.Input<Tensor>("BBoxes");
    auto* scores = ctx.Input<Tensor>("Scores");
    auto* anchors = ctx.Input<Tensor>("Anchors");
    auto* im_info = ctx.Input<LoDTensor>("ImInfo");
    auto* outs = ctx.Output<LoDTensor>("Out");

    auto score_dims = scores->dims();
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();

    std::vector<std::vector<std::vector<T>>> all_nmsed_out;
    std::vector<size_t> batch_starts = {0};
    int64_t batch_size = score_dims[0];
    int64_t box_dim = boxes->dims()[2];
    int64_t out_dim = box_dim + 2;
    for (int i = 0; i < batch_size; ++i) {
      int num_nmsed_out = 0;
      Tensor scores_slice = scores->Slice(i, i + 1);
      scores_slice.Resize({score_dims[1], score_dims[2]});
      Tensor boxes_slice = boxes->Slice(i, i + 1);
      boxes_slice.Resize({score_dims[1], box_dim});
      Tensor im_info_slice = im_info->Slice(i, i + 1);

      std::vector<std::vector<T>> nmsed_out;
      RetinanetDetectionOutput(ctx, scores_slice, boxes_slice, anchors,
                               im_info_slice, &nmsed_out, &num_nmsed_out);
      all_nmsed_out.push_back(nmsed_out);
      batch_starts.push_back(batch_starts.back() + num_nmsed_out);
    }

    int num_kept = batch_starts.back();
    if (num_kept == 0) {
      T* od = outs->mutable_data<T>({1, 1}, ctx.GetPlace());
      od[0] = -1;
      batch_starts = {0, 1};
    } else {
      outs->mutable_data<T>({num_kept, out_dim}, ctx.GetPlace());
      for (int i = 0; i < batch_size; ++i) {
        int64_t s = batch_starts[i];
        int64_t e = batch_starts[i + 1];
        if (e > s) {
          Tensor out = outs->Slice(s, e);
          MultiClassOutput(dev_ctx, all_nmsed_out[i], &out);
        }
      }
    }

    framework::LoD lod;
    lod.emplace_back(batch_starts);

    outs->set_lod(lod);
  }
};

class RetinanetDetectionOutputOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("BBoxes",
             "(Tensor) A 3-D Tensor with shape [N, M, 4] represents the "
             "predicted locations of M bounding boxes, N is the batch size. "
             "M is the number of bounding boxes from all FPN level. Each "
             "bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax].");
    AddInput("Scores",
             "(Tensor) A 3-D Tensor with shape [N, M, C] represents the "
             "predicted confidence predictions. N is the batch size, C is the "
             "class number, M is the number of bounding boxes from all FPN "
             "level. For each bounding box, there are total C scores.");
    AddInput("Anchors",
             "(Tensor) A 2-D Tensor with shape [M, 4] represents the locations "
             "of M anchor bboxes from all FPN level. "
             "Each bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax].");
    AddInput("ImInfo",
             "(LoDTensor) A 2-D LoDTensor with shape [N, 3] represents the "
             "image information. N is the batch size, each image information "
             "includes height, width and scale.");
    AddAttr<int>("min_level",
                 "The lowest level of FPN layer where the boxes come from.");
    AddAttr<int>("max_level",
                 "The highest level of FPN layer where the boxes come from.");
    AddAttr<float>("score_threshold",
                   "(float) "
                   "Threshold to filter out bounding boxes with a confidence "
                   "score.");
    AddAttr<int>("nms_top_k",
                 "(int64_t) "
                 "Maximum number of detections per FPN layer to be kept "
                 "according to the confidences before NMS.");
    AddAttr<float>("nms_threshold",
                   "(float) "
                   "The threshold to be used in NMS.");
    AddAttr<float>("nms_eta",
                   "(float) "
                   "The parameter for adaptive NMS.");
    AddAttr<int>("keep_top_k",
                 "(int64_t) "
                 "Number of total bboxes to be kept per image after NMS "
                 "step.");
    AddOutput("Out",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 6] represents the "
              "detections. Each row has 6 values: "
              "[label, confidence, xmin, ymin, xmax, ymax]"
              "No is the total number of detections in this mini-batch."
              "For each instance, "
              "the offsets in first dimension are called LoD, the number of "
              "offset is N + 1, if LoD[i + 1] - LoD[i] == 0, means there is "
              "no detected bbox.");
    AddComment(R"DOC(
This operator is to decode boxes and scores from each FPN layer and do
multi-class non maximum suppression (NMS) on merged predictions.

Firstly, input bounding box predictions are divided into box predictions per
FPN level, according to that the box number of the i-th level is
4**(max_level-i) times of that of the highest FPN level, where `max_level`
is the highest level of FPN layer where the boxes come from.

Next, top-scoring predictions per FPN layer are decoded with the anchor
information. This operator greedily selects a subset of detection bounding
boxes from each FPN layer that have high scores larger than score_threshold,
if providing this threshold, then selects the largest nms_top_k confidences
scores per FPN layer, if nms_top_k is larger than -1.
The Decoding schema described below:

ox = (pw * pxv * tx * + px) - tw / 2

oy = (ph * pyv * ty * + py) - th / 2

ow = exp(pwv * tw) * pw + tw / 2

oh = exp(phv * th) * ph + th / 2

where `tx`, `ty`, `tw`, `th` denote the predicted box's center coordinates, width
and height respectively. Similarly, `px`, `py`, `pw`, `ph` denote the
anchor's center coordinates, width and height. `pxv`, `pyv`, `pwv`,
`phv` denote the variance of the anchor box and `ox`, `oy`, `ow`, `oh` denote the
decoded coordinates, width and height. 

Then the top decoded prediction from all levels are merged followed by NMS.
In the NMS step, this operator pruns away boxes that have high IOU
(intersection over union) overlap with already selected boxes by adaptive
threshold NMS based on parameters of nms_threshold and nms_eta.
Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
per image if keep_top_k is larger than -1.
This operator support multi-class and batched inputs. It applying NMS
independently for each class. The outputs is a 2-D LoDTenosr, for each
image, the offsets in first dimension of LoDTensor are called LoD, the number
of offset is N + 1, where N is the batch size. If LoD[i + 1] - LoD[i] == 0,
means there is no detected bbox for this image. If there is no detected boxes
for all images, all the elements in LoD are set to {1}, and the Out only 
contains one value which is -1.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(retinanet_detection_output, ops::RetinanetDetectionOutputOp,
                  ops::RetinanetDetectionOutputOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(retinanet_detection_output,
                       ops::RetinanetDetectionOutputKernel<float>,
                       ops::RetinanetDetectionOutputKernel<double>);
