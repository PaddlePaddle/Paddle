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

#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

static void AppendProposals(Tensor *dst, int64_t offset, const Tensor &src) {
  auto *out_data = dst->data<void>();
  auto *to_add_data = src.data<void>();
  size_t size_of_t = framework::SizeOfType(src.type());
  offset *= size_of_t;
  std::memcpy(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(out_data) + offset),
      to_add_data, src.numel() * size_of_t);
}

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

    ctx->SetOutputDim("RpnRois", {-1, 4});
    ctx->SetOutputDim("RpnRoiProbs", {-1, 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Anchors")->type(),
                                   ctx.device_context());
  }
};

template <class T>
static inline void BoxCoder(const platform::DeviceContext &ctx,
                            Tensor *all_anchors, Tensor *bbox_deltas,
                            Tensor *variances, Tensor *proposals) {
  T *proposals_data = proposals->mutable_data<T>(ctx.GetPlace());

  int64_t row = all_anchors->dims()[0];
  int64_t len = all_anchors->dims()[1];

  auto *bbox_deltas_data = bbox_deltas->data<T>();
  auto *anchor_data = all_anchors->data<T>();
  const T *variances_data = nullptr;
  if (variances) {
    variances_data = variances->data<T>();
  }

  for (int64_t i = 0; i < row; ++i) {
    T anchor_width = anchor_data[i * len + 2] - anchor_data[i * len] + 1.0;
    T anchor_height = anchor_data[i * len + 3] - anchor_data[i * len + 1] + 1.0;

    T anchor_center_x = anchor_data[i * len] + 0.5 * anchor_width;
    T anchor_center_y = anchor_data[i * len + 1] + 0.5 * anchor_height;

    T bbox_center_x = 0, bbox_center_y = 0;
    T bbox_width = 0, bbox_height = 0;

    if (variances) {
      bbox_center_x =
          variances_data[i * len] * bbox_deltas_data[i * len] * anchor_width +
          anchor_center_x;
      bbox_center_y = variances_data[i * len + 1] *
                          bbox_deltas_data[i * len + 1] * anchor_height +
                      anchor_center_y;
      bbox_width = std::exp(std::min<T>(variances_data[i * len + 2] *
                                            bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(variances_data[i * len + 3] *
                                             bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    } else {
      bbox_center_x =
          bbox_deltas_data[i * len] * anchor_width + anchor_center_x;
      bbox_center_y =
          bbox_deltas_data[i * len + 1] * anchor_height + anchor_center_y;
      bbox_width = std::exp(std::min<T>(bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    }

    proposals_data[i * len] = bbox_center_x - bbox_width / 2;
    proposals_data[i * len + 1] = bbox_center_y - bbox_height / 2;
    proposals_data[i * len + 2] = bbox_center_x + bbox_width / 2 - 1;
    proposals_data[i * len + 3] = bbox_center_y + bbox_height / 2 - 1;
  }
  // return proposals;
}

template <class T>
static inline void ClipTiledBoxes(const platform::DeviceContext &ctx,
                                  const Tensor &im_info, Tensor *boxes) {
  T *boxes_data = boxes->mutable_data<T>(ctx.GetPlace());
  const T *im_info_data = im_info.data<T>();
  T zero(0);
  for (int64_t i = 0; i < boxes->numel(); ++i) {
    if (i % 4 == 0) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), zero);
    } else if (i % 4 == 1) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[0] - 1), zero);
    } else if (i % 4 == 2) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), zero);
    } else {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[0] - 1), zero);
    }
  }
}

template <class T>
static inline void FilterBoxes(const platform::DeviceContext &ctx,
                               Tensor *boxes, float min_size,
                               const Tensor &im_info, Tensor *keep) {
  const T *im_info_data = im_info.data<T>();
  T *boxes_data = boxes->mutable_data<T>(ctx.GetPlace());
  T im_scale = im_info_data[2];
  keep->Resize({boxes->dims()[0]});
  min_size = std::max(min_size, 1.0f);
  int *keep_data = keep->mutable_data<int>(ctx.GetPlace());

  int keep_len = 0;
  for (int i = 0; i < boxes->dims()[0]; ++i) {
    T ws = boxes_data[4 * i + 2] - boxes_data[4 * i] + 1;
    T hs = boxes_data[4 * i + 3] - boxes_data[4 * i + 1] + 1;
    T ws_origin_scale =
        (boxes_data[4 * i + 2] - boxes_data[4 * i]) / im_scale + 1;
    T hs_origin_scale =
        (boxes_data[4 * i + 3] - boxes_data[4 * i + 1]) / im_scale + 1;
    T x_ctr = boxes_data[4 * i] + ws / 2;
    T y_ctr = boxes_data[4 * i + 1] + hs / 2;
    if (ws_origin_scale >= min_size && hs_origin_scale >= min_size &&
        x_ctr <= im_info_data[1] && y_ctr <= im_info_data[0]) {
      keep_data[keep_len++] = i;
    }
  }
  keep->Resize({keep_len});
}

template <class T>
static inline std::vector<std::pair<T, int>> GetSortedScoreIndex(
    const std::vector<T> &scores) {
  std::vector<std::pair<T, int>> sorted_indices;
  sorted_indices.reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    sorted_indices.emplace_back(scores[i], i);
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
                   [](const std::pair<T, int> &a, const std::pair<T, int> &b) {
                     return a.first < b.first;
                   });
  return sorted_indices;
}

template <class T>
static inline T BBoxArea(const T *box, bool normalized) {
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
static inline T JaccardOverlap(const T *box1, const T *box2, bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    const T inter_w = std::max(T(0), inter_xmax - inter_xmin + 1);
    const T inter_h = std::max(T(0), inter_ymax - inter_ymin + 1);
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <typename T>
static inline Tensor VectorToTensor(const std::vector<T> &selected_indices,
                                    int selected_num) {
  Tensor keep_nms;
  keep_nms.Resize({selected_num});
  auto *keep_data = keep_nms.mutable_data<T>(platform::CPUPlace());
  for (int i = 0; i < selected_num; ++i) {
    keep_data[i] = selected_indices[i];
  }
  return keep_nms;
}

template <class T>
static inline Tensor NMS(const platform::DeviceContext &ctx, Tensor *bbox,
                         Tensor *scores, T nms_threshold, float eta) {
  PADDLE_ENFORCE_NOT_NULL(bbox);
  int64_t num_boxes = bbox->dims()[0];
  // 4: [xmin ymin xmax ymax]
  int64_t box_size = bbox->dims()[1];

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores->data<T>(), num_boxes, scores_data.begin());
  std::vector<std::pair<T, int>> sorted_indices =
      GetSortedScoreIndex<T>(scores_data);

  std::vector<int> selected_indices;
  int selected_num = 0;
  T adaptive_threshold = nms_threshold;
  const T *bbox_data = bbox->data<T>();
  while (sorted_indices.size() != 0) {
    int idx = sorted_indices.back().second;
    bool flag = true;
    for (int kept_idx : selected_indices) {
      if (flag) {
        T overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                      bbox_data + kept_idx * box_size, false);
        flag = (overlap <= adaptive_threshold);
      } else {
        break;
      }
    }
    if (flag) {
      selected_indices.push_back(idx);
      ++selected_num;
    }
    sorted_indices.erase(sorted_indices.end() - 1);
    if (flag && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
  return VectorToTensor(selected_indices, selected_num);
}

template <typename T>
class GenerateProposalsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *scores = context.Input<Tensor>("Scores");
    auto *bbox_deltas = context.Input<Tensor>("BboxDeltas");
    auto *im_info = context.Input<Tensor>("ImInfo");
    auto anchors = detail::Ref(context.Input<Tensor>("Anchors"),
                               "Cannot find input Anchors(%s) in scope",
                               context.Inputs("Anchors")[0]);
    auto variances = detail::Ref(context.Input<Tensor>("Variances"),
                                 "Cannot find input Variances(%s) in scope",
                                 context.Inputs("Variances")[0]);

    auto *rpn_rois = context.Output<LoDTensor>("RpnRois");
    auto *rpn_roi_probs = context.Output<LoDTensor>("RpnRoiProbs");

    int pre_nms_top_n = context.Attr<int>("pre_nms_topN");
    int post_nms_top_n = context.Attr<int>("post_nms_topN");
    float nms_thresh = context.Attr<float>("nms_thresh");
    float min_size = context.Attr<float>("min_size");
    float eta = context.Attr<float>("eta");

    auto &dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();

    auto &scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto &bbox_dim = bbox_deltas->dims();
    int64_t c_bbox = bbox_dim[1];
    int64_t h_bbox = bbox_dim[2];
    int64_t w_bbox = bbox_dim[3];

    rpn_rois->mutable_data<T>({bbox_deltas->numel() / 4, 4},
                              context.GetPlace());
    rpn_roi_probs->mutable_data<T>({scores->numel(), 1}, context.GetPlace());

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.mutable_data<T>({num, h_bbox, w_bbox, c_bbox},
                                     dev_ctx.GetPlace());
    scores_swap.mutable_data<T>({num, h_score, w_score, c_score},
                                dev_ctx.GetPlace());

    math::Transpose<platform::CPUDeviceContext, T, 4> trans;
    std::vector<int> axis = {0, 2, 3, 1};
    trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
    trans(dev_ctx, *scores, &scores_swap, axis);

    framework::LoD lod;
    lod.resize(1);
    auto &lod0 = lod[0];
    lod0.push_back(0);
    anchors.Resize({anchors.numel() / 4, 4});
    variances.Resize({variances.numel() / 4, 4});

    int64_t num_proposals = 0;
    for (int64_t i = 0; i < num; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
      Tensor scores_slice = scores_swap.Slice(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
      scores_slice.Resize({h_score * w_score * c_score, 1});

      std::pair<Tensor, Tensor> tensor_pair =
          ProposalForOneImage(dev_ctx, im_info_slice, anchors, variances,
                              bbox_deltas_slice, scores_slice, pre_nms_top_n,
                              post_nms_top_n, nms_thresh, min_size, eta);
      Tensor &proposals = tensor_pair.first;
      Tensor &scores = tensor_pair.second;

      AppendProposals(rpn_rois, 4 * num_proposals, proposals);
      AppendProposals(rpn_roi_probs, num_proposals, scores);
      num_proposals += proposals.dims()[0];
      lod0.push_back(num_proposals);
    }
    rpn_rois->set_lod(lod);
    rpn_roi_probs->set_lod(lod);
    rpn_rois->Resize({num_proposals, 4});
    rpn_roi_probs->Resize({num_proposals, 1});
  }

  std::pair<Tensor, Tensor> ProposalForOneImage(
      const platform::CPUDeviceContext &ctx, const Tensor &im_info_slice,
      const Tensor &anchors, const Tensor &variances,
      const Tensor &bbox_deltas_slice,  // [M, 4]
      const Tensor &scores_slice,       // [N, 1]
      int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size,
      float eta) const {
    auto *scores_data = scores_slice.data<T>();

    // Sort index
    Tensor index_t;
    index_t.Resize({scores_slice.numel()});
    int *index = index_t.mutable_data<int>(ctx.GetPlace());
    for (int i = 0; i < scores_slice.numel(); ++i) {
      index[i] = i;
    }
    auto compare = [scores_data](const int64_t &i, const int64_t &j) {
      return scores_data[i] > scores_data[j];
    };

    if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_slice.numel()) {
      std::sort(index, index + scores_slice.numel(), compare);
    } else {
      std::nth_element(index, index + pre_nms_top_n,
                       index + scores_slice.numel(), compare);
      index_t.Resize({pre_nms_top_n});
    }

    Tensor scores_sel, bbox_sel, anchor_sel, var_sel;
    scores_sel.mutable_data<T>({index_t.numel(), 1}, ctx.GetPlace());
    bbox_sel.mutable_data<T>({index_t.numel(), 4}, ctx.GetPlace());
    anchor_sel.mutable_data<T>({index_t.numel(), 4}, ctx.GetPlace());
    var_sel.mutable_data<T>({index_t.numel(), 4}, ctx.GetPlace());

    CPUGather<T>(ctx, scores_slice, index_t, &scores_sel);
    CPUGather<T>(ctx, bbox_deltas_slice, index_t, &bbox_sel);
    CPUGather<T>(ctx, anchors, index_t, &anchor_sel);
    CPUGather<T>(ctx, variances, index_t, &var_sel);

    Tensor proposals;
    proposals.mutable_data<T>({index_t.numel(), 4}, ctx.GetPlace());
    BoxCoder<T>(ctx, &anchor_sel, &bbox_sel, &var_sel, &proposals);

    ClipTiledBoxes<T>(ctx, im_info_slice, &proposals);

    Tensor keep;
    FilterBoxes<T>(ctx, &proposals, min_size, im_info_slice, &keep);

    Tensor scores_filter;
    bbox_sel.mutable_data<T>({keep.numel(), 4}, ctx.GetPlace());
    scores_filter.mutable_data<T>({keep.numel(), 1}, ctx.GetPlace());
    CPUGather<T>(ctx, proposals, keep, &bbox_sel);
    CPUGather<T>(ctx, scores_sel, keep, &scores_filter);
    if (nms_thresh <= 0) {
      return std::make_pair(bbox_sel, scores_filter);
    }

    Tensor keep_nms = NMS<T>(ctx, &bbox_sel, &scores_filter, nms_thresh, eta);

    if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
      keep_nms.Resize({post_nms_top_n});
    }

    proposals.mutable_data<T>({keep_nms.numel(), 4}, ctx.GetPlace());
    scores_sel.mutable_data<T>({keep_nms.numel(), 1}, ctx.GetPlace());
    CPUGather<T>(ctx, bbox_sel, keep_nms, &proposals);
    CPUGather<T>(ctx, scores_filter, keep_nms, &scores_sel);

    return std::make_pair(proposals, scores_sel);
  }
};

class GenerateProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Scores",
             "(Tensor) The scores from conv is in shape (N, A, H, W), "
             "N is batch size, A is number of anchors, "
             "H and W are height and width of the feature map");
    AddInput("BboxDeltas",
             "(Tensor) Bounding box deltas from conv is in "
             "shape (N, 4*A, H, W).");
    AddInput("ImInfo",
             "(Tensor) Information for image reshape is in shape (N, 3), "
             "in format (height, width, scale)");
    AddInput("Anchors",
             "(Tensor) Bounding box anchors from anchor_generator_op "
             "is in shape (A, H, W, 4).");
    AddInput("Variances",
             "(Tensor) Bounding box variances with same shape as `Anchors`.");

    AddOutput("RpnRois",
              "(LoDTensor), Output proposals with shape (rois_num, 4).");
    AddOutput("RpnRoiProbs",
              "(LoDTensor) Scores of proposals with shape (rois_num, 1).");
    AddAttr<int>("pre_nms_topN",
                 "Number of top scoring RPN proposals to keep before "
                 "applying NMS.");
    AddAttr<int>("post_nms_topN",
                 "Number of top scoring RPN proposals to keep after "
                 "applying NMS");
    AddAttr<float>("nms_thresh", "NMS threshold used on RPN proposals.");
    AddAttr<float>("min_size",
                   "Proposal height and width both need to be greater "
                   "than this min_size.");
    AddAttr<float>("eta", "The parameter for adaptive NMS.");
    AddComment(R"DOC(
This operator Generate bounding box proposals for Faster RCNN.
The propoasls are generated for a list of images based on image
score 'Scores', bounding box regression result 'BboxDeltas' as
well as predefined bounding box shapes 'anchors'. Greedy
non-maximum suppression is applied to generate the final bounding
boxes.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(generate_proposals, ops::GenerateProposalsOp,
                  ops::GenerateProposalsOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(generate_proposals, ops::GenerateProposalsKernel<float>,
                       ops::GenerateProposalsKernel<double>);
