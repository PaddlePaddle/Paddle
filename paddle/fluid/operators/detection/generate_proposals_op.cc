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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

struct AppendProposalsFunctor {
  LoDTensor *out_;
  int64_t offset_;
  Tensor *to_add_;

  AppendProposalsFunctor(LoDTensor *out, int64_t offset, Tensor *to_add)
      : out_(out), offset_(offset), to_add_(to_add) {}

  template <typename T>
  void operator()() const {
    auto *out_data = out_->data<T>();
    auto *to_add_data = to_add_->data<T>();
    memcpy(out_data + offset_, to_add_data, to_add_->numel() * sizeof(T));
  }
};

class GenerateProposalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Scores"), "Input(Scores) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("BboxDeltas"),
                   "Input(BboxDeltas) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("ImInfo"), "Input(ImInfo) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Anchors"),
                   "Input(Anchors) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Variances"),
                   "Input(Variances) shouldn't be null.");

    auto scores_dims = ctx->GetInputDim("Scores");
    auto bbox_deltas_dims = ctx->GetInputDim("BboxDeltas");
    auto im_info_dims = ctx->GetInputDim("ImInfo");
    auto anchors_dims = ctx->GetInputDim("Anchors");
    auto variances_dims = ctx->GetInputDim("Variances");

    ctx->SetOutputDim("RpnRois", anchors_dims);
    ctx->SetOutputDim("RpnRoiProbs", scores_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Anchors")->type()),
        platform::CPUPlace());
  }
};

template <typename DeviceContext>
void Trans(const DeviceContext &ctx, const Tensor &in_tensor,
           Tensor *out_tensor, const std::vector<int> &axis) {
  math::Transpose<DeviceContext, float, 3> trans;
  trans(ctx, in_tensor, out_tensor, axis);
  out_tensor->Resize(out_tensor->dims());
}

void Gather(const platform::DeviceContext &ctx, const Tensor &in,
            const Tensor &index, Tensor *out) {
  out->Resize({index.numel(), 1});
  CPUGather<float>(ctx, in, index, out);
  out->Resize({index.numel(), in.dims()[1]});
}

Tensor BoxCoder(const platform::DeviceContext &ctx, Tensor *all_anchors,
                Tensor *bbox_deltas, Tensor *variances) {
  std::vector<int64_t> bbox_deltas_dims =
      framework::vectorize(bbox_deltas->dims());
  framework::DDim bbox_deltas_shape = framework::make_ddim(bbox_deltas_dims);
  bbox_deltas->Resize(bbox_deltas_shape);

  Tensor proposals;
  proposals.Resize(all_anchors->dims());
  proposals.mutable_data(ctx.GetPlace(), all_anchors->type());
  float *proposals_data = proposals.mutable_data<float>(ctx.GetPlace());

  int64_t row = all_anchors->dims()[0];
  int64_t len = all_anchors->dims()[1];

  auto *bbox_deltas_data = bbox_deltas->data<float>();
  auto *anchor_data = all_anchors->data<float>();
  const float *variances_data = nullptr;
  if (variances) {
    variances_data = variances->data<float>();
  }

  for (int64_t i = 0; i < row; ++i) {
    float anchor_width = anchor_data[i * len + 2] - anchor_data[i * len];
    float anchor_height = anchor_data[i * len + 3] - anchor_data[i * len + 1];

    float anchor_center_x =
        (anchor_data[i * len + 2] + anchor_data[i * len]) / 2;
    float anchor_center_y =
        (anchor_data[i * len + 3] + anchor_data[i * len + 1]) / 2;

    float bbox_center_x = 0, bbox_center_y = 0;
    float bbox_width = 0, bbox_height = 0;

    if (variances) {
      bbox_center_x =
          variances_data[i * len] * bbox_deltas_data[i * len] * anchor_width +
          anchor_center_x;
      bbox_center_y = variances_data[i * len + 1] *
                          bbox_deltas_data[i * len + 1] * anchor_height +
                      anchor_center_y;
      bbox_width = std::exp(variances_data[i * len + 2] *
                            bbox_deltas_data[i * len + 2]) *
                   anchor_width;
      bbox_height = std::exp(variances_data[i * len + 3] *
                             bbox_deltas_data[i * len + 3]) *
                    anchor_height;
    } else {
      bbox_center_x =
          bbox_deltas_data[i * len] * anchor_width + anchor_center_x;
      bbox_center_y =
          bbox_deltas_data[i * len + 1] * anchor_height + anchor_center_y;
      bbox_width = std::exp(bbox_deltas_data[i * len + 2]) * anchor_width;
      bbox_height = std::exp(bbox_deltas_data[i * len + 3]) * anchor_height;
    }

    proposals_data[i * len] = bbox_center_x - bbox_width / 2;
    proposals_data[i * len + 1] = bbox_center_y - bbox_height / 2;
    proposals_data[i * len + 2] = bbox_center_x + bbox_width / 2;
    proposals_data[i * len + 3] = bbox_center_y + bbox_height / 2;
  }
  return proposals;
}

void ClipTiledBoxes(const platform::DeviceContext &ctx, Tensor *boxes,
                    Tensor *im_info) {
  float *boxes_data = boxes->mutable_data<float>(ctx.GetPlace());
  float *im_info_data = im_info->mutable_data<float>(ctx.GetPlace());
  for (int64_t i = 0; i < boxes->numel(); ++i) {
    if (i % 4 == 0) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), 0.0f);
    } else if (i % 4 == 1) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[0] - 1), 0.0f);
    } else if (i % 4 == 2) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), 0.0f);
    } else {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[0] - 1), 0.0f);
    }
  }
}

Tensor FilterBoxes(const platform::DeviceContext &ctx, Tensor *boxes,
                   float min_size, Tensor *im_info) {
  float *im_info_data = im_info->mutable_data<float>(ctx.GetPlace());
  float *boxes_data = boxes->mutable_data<float>(ctx.GetPlace());
  min_size *= im_info_data[2];
  Tensor keep;
  keep.Resize({boxes->dims()[0]});
  int *keep_data = keep.mutable_data<int>(ctx.GetPlace());

  int keep_len = 0;
  for (int i = 0; i < keep.numel(); ++i) {
    float ws = boxes_data[4 * i + 2] - boxes_data[4 * i] + 1;
    float hs = boxes_data[4 * i + 3] - boxes_data[4 * i + 1] + 1;
    float x_ctr = boxes_data[4 * i] + ws / 2;
    float y_ctr = boxes_data[4 * i + 1] + hs / 2;
    if (ws >= min_size && hs >= min_size && x_ctr <= im_info_data[1] &&
        y_ctr <= im_info_data[0]) {
      keep_data[keep_len++] = i;
    }
  }
  keep.Resize({keep_len});
  return keep;
}

bool SortScorePairDescend(const std::pair<float, int> &pair1,
                          const std::pair<float, int> &pair2) {
  return pair1.first > pair2.first;
}

template <class T>
void GetMaxScoreIndex(const std::vector<T> &scores,
                      std::vector<std::pair<T, int>> *sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    sorted_indices->push_back(std::make_pair(scores[i], i));
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend);
}

template <class T>
T BBoxArea(const T *box, const bool normalized) {
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
T JaccardOverlap(const T *box1, const T *box2, const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    const T inter_w = inter_xmax - inter_xmin;
    const T inter_h = inter_ymax - inter_ymin;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <class T>
Tensor NMS(const platform::DeviceContext &ctx, Tensor *bbox, Tensor *scores,
           const T nms_threshold, const float eta) {
  PADDLE_ENFORCE_NOT_NULL(bbox);
  int64_t num_boxes = bbox->dims()[0];
  // 4: [xmin ymin xmax ymax]
  int64_t box_size = bbox->dims()[1];

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores->data<T>(), num_boxes, scores_data.begin());
  std::vector<std::pair<T, int>> sorted_indices;
  GetMaxScoreIndex<T>(scores_data, &sorted_indices);

  std::vector<int> selected_indices;
  int selected_num = 0;
  T adaptive_threshold = nms_threshold;
  const T *bbox_data = bbox->data<T>();
  bool flag;
  while (sorted_indices.size() != 0) {
    int idx = sorted_indices.front().second;
    flag = true;
    for (size_t k = 0; k < selected_indices.size(); ++k) {
      if (flag) {
        const int kept_idx = selected_indices[k];
        T overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                      bbox_data + kept_idx * box_size, false);
        flag = (overlap <= adaptive_threshold);
      } else {
        break;
      }
    }
    if (flag) {
      selected_indices.push_back(idx);
      selected_num++;
    }
    sorted_indices.erase(sorted_indices.begin());
    if (flag && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
  Tensor keep_nms;
  keep_nms.Resize({selected_num});
  int *keep_data = keep_nms.mutable_data<int>(ctx.GetPlace());
  for (int i = 0; i < selected_num; ++i) {
    keep_data[i] = selected_indices[i];
  }
  return keep_nms;
}

template <typename DeviceContext, typename T>
class GenerateProposalsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *scores = context.Input<LoDTensor>("Scores");
    auto *bbox_deltas = context.Input<LoDTensor>("BboxDeltas");
    auto *im_info = context.Input<LoDTensor>("ImInfo");
    auto *anchors = context.Input<LoDTensor>("Anchors");
    auto *variances = context.Input<LoDTensor>("Variances");

    auto *rpn_rois = context.Output<LoDTensor>("RpnRois");
    auto *rpn_roi_probs = context.Output<LoDTensor>("RpnRoiProbs");

    rpn_rois->mutable_data(context.GetPlace(), anchors->type());
    rpn_roi_probs->mutable_data(context.GetPlace(), scores->type());
    int pre_nms_topN = context.Attr<int>("pre_nms_topN");
    int post_nms_topN = context.Attr<int>("post_nms_topN");
    float nms_thresh = context.Attr<float>("nms_thresh");
    float min_size = context.Attr<float>("min_size");
    float eta = context.Attr<float>("eta");

    // paddle::framework::Scope &local_scope = scope.NewScope();
    auto &dev_ctx = context.template device_context<DeviceContext>();

    framework::LoD lod;
    std::vector<size_t> lod0(1, 0);

    int64_t num_images = scores->dims()[0];
    int64_t num_proposals = 0;
    for (int64_t i = 0; i < num_images; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas->Slice(i, i + 1);
      Tensor scores_slice = scores->Slice(i, i + 1);
      std::pair<Tensor, Tensor> tensor_pair =
          ProposalForOneImage(dev_ctx, &im_info_slice, *anchors, *variances,
                              &bbox_deltas_slice, &scores_slice, pre_nms_topN,
                              post_nms_topN, nms_thresh, min_size, eta);
      Tensor proposals = tensor_pair.first;
      Tensor scores = tensor_pair.second;

      framework::VisitDataType(
          framework::ToDataType(rpn_rois->type()),
          AppendProposalsFunctor(rpn_rois, 4 * num_proposals, &proposals));
      framework::VisitDataType(
          framework::ToDataType(rpn_roi_probs->type()),
          AppendProposalsFunctor(rpn_roi_probs, num_proposals, &scores));

      num_proposals += proposals.dims()[0];
      lod0.emplace_back(num_proposals);
    }

    lod.emplace_back(lod0);
    rpn_rois->set_lod(lod);
    rpn_roi_probs->set_lod(lod);
    rpn_rois->Resize({num_proposals, 4});
    rpn_roi_probs->Resize({num_proposals, 1});
  }

  std::pair<Tensor, Tensor> ProposalForOneImage(
      const DeviceContext &ctx, Tensor *im_info_slice, const Tensor &anchors,
      const Tensor &variances, Tensor *bbox_deltas_slice, Tensor *scores_slice,
      int pre_nms_topN, int post_nms_topN, float nms_thresh, float min_size,
      float eta) const {
    Tensor bbox_deltas, bbox_deltas_swap;
    Tensor scores, scores_swap;

    framework::DDim bbox_deltas_shape = framework::slice_ddim(
        bbox_deltas_slice->dims(), 1, bbox_deltas_slice->dims().size());
    framework::DDim scores_shape = framework::slice_ddim(
        scores_slice->dims(), 1, scores_slice->dims().size());

    bbox_deltas = *bbox_deltas_slice;
    scores = *scores_slice;
    bbox_deltas.Resize(bbox_deltas_shape);
    scores.Resize(scores_shape);

    bbox_deltas_swap.Resize(bbox_deltas_shape);
    scores_swap.Resize(scores_shape);
    bbox_deltas_swap.mutable_data(ctx.GetPlace(), bbox_deltas.type());
    scores_swap.mutable_data(ctx.GetPlace(), scores.type());

    // Transpose bbox_deltas
    std::vector<int> axis = {1, 2, 0};
    Trans<DeviceContext>(ctx, bbox_deltas, &bbox_deltas_swap, axis);

    // Transpose scores
    Trans<DeviceContext>(ctx, scores, &scores_swap, axis);

    scores_swap.Resize({scores_swap.numel(), 1});
    auto *scores_data = scores_swap.mutable_data<float>(ctx.GetPlace());

    // Sort index
    Tensor index_t;
    index_t.Resize({scores_swap.numel()});
    int *index = index_t.mutable_data<int>(ctx.GetPlace());
    for (int64_t i = 0; i < scores.numel(); ++i) {
      index[i] = i;
    }
    std::function<bool(const int64_t &, const int64_t &)> compare =
        [scores_data](const int64_t &i, const int64_t &j) {
          return scores_data[i] > scores_data[j];
        };
    if (pre_nms_topN <= 0 || pre_nms_topN >= scores_swap.numel()) {
      std::sort(index, index + scores.numel(), compare);
    } else {
      std::nth_element(index, index + pre_nms_topN, index + scores.numel(),
                       compare);
      index_t.Resize({pre_nms_topN});
    }

    Gather(ctx, scores_swap, index_t, &scores);
    bbox_deltas_swap.Resize({bbox_deltas_swap.numel() / 4, 4});
    Gather(ctx, bbox_deltas_swap, index_t, &bbox_deltas);

    Tensor all_anchors, all_anchors_swap;
    all_anchors_swap = anchors;
    all_anchors_swap.Resize({all_anchors_swap.numel() / 4, 4});
    all_anchors.Resize(all_anchors_swap.dims());
    all_anchors.mutable_data(ctx.GetPlace(), all_anchors_swap.type());

    Gather(ctx, all_anchors_swap, index_t, &all_anchors);

    Tensor box_var;
    box_var = variances;
    box_var.Resize({box_var.numel() / 4, 4});
    box_var.mutable_data(ctx.GetPlace(), box_var.type());

    Tensor proposals = BoxCoder(ctx, &all_anchors, &bbox_deltas, &box_var);

    ClipTiledBoxes(ctx, &proposals, im_info_slice);

    Tensor keep = FilterBoxes(ctx, &proposals, min_size, im_info_slice);

    Tensor proposals_swap;
    proposals_swap.Resize(proposals.dims());
    proposals_swap.mutable_data(ctx.GetPlace(), proposals.type());

    Gather(ctx, proposals, keep, &proposals_swap);
    Gather(ctx, scores, keep, &scores_swap);

    if (nms_thresh <= 0) {
      return std::make_pair(proposals_swap, scores_swap);
    }

    Tensor keep_nms =
        NMS<T>(ctx, &proposals_swap, &scores_swap, nms_thresh, eta);

    if (post_nms_topN > 0 && post_nms_topN < keep_nms.numel()) {
      keep_nms.Resize({post_nms_topN});
    }
    Gather(ctx, proposals_swap, keep_nms, &proposals);
    Gather(ctx, scores_swap, keep_nms, &scores);

    return std::make_pair(proposals, scores);
  }
};

class GenerateProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Scores", "The scores of anchors should be foreground.");
    AddInput("BboxDeltas", "bbox_deltas.");
    AddInput("ImInfo", "Information for image reshape.");
    AddInput("Anchors", "All anchors.");
    AddInput("Variances", " variances");

    AddOutput("RpnRois", "Anchors.");
    AddOutput("RpnRoiProbs", "Anchors.");
    AddAttr<int>("pre_nms_topN", "pre_nms_topN");
    AddAttr<int>("post_nms_topN", "post_nms_topN");
    AddAttr<float>("nms_thresh", "nms_thres");
    AddAttr<float>("min_size", "min size");
    AddAttr<float>("eta", "eta");
    AddComment(R"DOC(
Generate Proposals Operator.


)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(generate_proposals, ops::GenerateProposalsOp,
                  ops::GenerateProposalsOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    generate_proposals,
    ops::GenerateProposalsKernel<paddle::platform::CPUDeviceContext, float>);
