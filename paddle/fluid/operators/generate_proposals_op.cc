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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

struct DyDataTypeVisitor {
  const platform::Place place_;
  LoDTensor *in_;
  DyDataTypeVisitor(const platform::Place &place, LoDTensor *in)
      : place_(place), in_(in) {}

  template <typename T>
  T *operator()() {
    auto *p = in_->mutable_data<T>(place_);
    return p;
  }
};

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

class GenerateProposalsOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
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
};

void Transpose(framework::Scope *scope, const platform::Place &place,
               const Tensor &in_tensor, Tensor *out_tensor,
               const framework::AttributeMap &attrs) {
  std::string in_name, out_name;
  scope->Var(&in_name)->GetMutable<LoDTensor>()->ShareDataWith(in_tensor);
  scope->Var(&out_name)->GetMutable<LoDTensor>()->ShareDataWith(*out_tensor);

  auto transpose_op = framework::OpRegistry::CreateOp(
      "transpose", {{"X", {in_name}}}, {{"Out", {out_name}}}, attrs);
  transpose_op->Run(*scope, place);

  out_tensor->Resize(scope->FindVar(out_name)->GetMutable<LoDTensor>()->dims());
}

void Gather(framework::Scope *scope, const platform::Place &place,
            const Tensor &in, const Tensor &index, Tensor *out) {
  std::string in_name, index_name, out_name;
  scope->Var(&in_name)->GetMutable<LoDTensor>()->ShareDataWith(in);
  scope->Var(&index_name)->GetMutable<LoDTensor>()->ShareDataWith(index);
  scope->Var(&out_name)->GetMutable<LoDTensor>()->ShareDataWith(*out);

  framework::AttributeMap attrs;
  auto gather_op = framework::OpRegistry::CreateOp(
      "gather", {{"X", {in_name}}, {"Index", {index_name}}},
      {{"Out", {out_name}}}, attrs);
  gather_op->Run(*scope, place);

  out->Resize(scope->FindVar(out_name)->GetMutable<LoDTensor>()->dims());
}

Tensor BoxCoder(framework::Scope *scope, const platform::Place &place,
                Tensor *all_anchors, Tensor *bbox_deltas,
                const Tensor &variances, std::string code_type,
                bool box_normalized) {
  std::vector<int64_t> bbox_deltas_dims =
      framework::vectorize(bbox_deltas->dims());
  bbox_deltas_dims.emplace(bbox_deltas_dims.begin(), 1);
  framework::DDim bbox_deltas_shape = framework::make_ddim(bbox_deltas_dims);
  bbox_deltas->Resize(bbox_deltas_shape);

  Tensor proposals;
  proposals.Resize(all_anchors->dims());
  proposals.mutable_data(place, all_anchors->type());

  Tensor variances_swap = variances;
  variances_swap.Resize({variances_swap.numel() / 4, 4});

  framework::AttributeMap attrs;
  attrs["code_type"] = code_type;
  attrs["box_normalized"] = box_normalized;

  std::string all_anchors_name, bbox_deltas_name, variances_name,
      proposals_name;
  scope->Var(&all_anchors_name)
      ->GetMutable<LoDTensor>()
      ->ShareDataWith(*all_anchors);
  scope->Var(&bbox_deltas_name)
      ->GetMutable<LoDTensor>()
      ->ShareDataWith(*bbox_deltas);
  scope->Var(&variances_name)
      ->GetMutable<LoDTensor>()
      ->ShareDataWith(variances_swap);
  scope->Var(&proposals_name)
      ->GetMutable<LoDTensor>()
      ->ShareDataWith(proposals);

  auto box_coder_op = framework::OpRegistry::CreateOp(
      "box_coder", {{"PriorBox", {all_anchors_name}},
                    {"PriorBoxVar", {variances_name}},
                    {"TargetBox", {bbox_deltas_name}}},
      {{"OutputBox", {proposals_name}}}, attrs);
  box_coder_op->Run(*scope, place);

  bbox_deltas_shape =
      framework::slice_ddim(bbox_deltas->dims(), 1, bbox_deltas->dims().size());
  bbox_deltas->Resize(bbox_deltas_shape);

  return proposals;
}

void ClipTiledBoxes(const platform::Place &place, Tensor *boxes,
                    Tensor *im_info) {
  float *boxes_data = boxes->mutable_data<float>(place);
  float *im_info_data = im_info->mutable_data<float>(place);
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

Tensor FilterBoxes(const platform::Place &place, Tensor *boxes, float min_size,
                   Tensor *im_info) {
  float *im_info_data = im_info->mutable_data<float>(place);
  float *boxes_data = boxes->mutable_data<float>(place);
  min_size *= im_info_data[2];
  Tensor keep;
  keep.Resize({boxes->dims()[0]});
  int *keep_data = keep.mutable_data<int>(place);

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

/*
void NMS(framework::Scope &scope,
           const platform::Place &place,
           Tensor &proposals,
           Tensor &scores,
           float nms_threshold,
           float score_threshold,
           int nms_top_k,
           float nms_eta,
           Tensor &keep) {
  framework::AttributeMap attrs;
  attrs["score_threshold"] = score_threshold;
  attrs["nms_threshold"] = nms_threshold;
  attrs["nms_eta"] = nms_eta;

  std::string proposals_name, scores_name, variances_name;
  scope.Var(&proposals_name)->GetMutable<LoDTensor>()
      ->ShareDataWith(proposals);
  scope.Var(&scores_name)->GetMutable<LoDTensor>()
      ->ShareDataWith(scores);
  scope.Var(&variances_name)->GetMutable<LoDTensor>()
      ->ShareDataWith(variances);

  auto multiclass_nms_op = framework::OpRegistry::CreateOp(
      "multiclass_nms", {{"BBoxes", {proposals_name}},
                    {"Scores", {scores_name}}}, attrs);
  multiclass_nms_op->Run(scope, place);
}
*/

std::pair<Tensor, Tensor> ProposalForOneImage(
    framework::Scope *scope, const platform::Place &place,
    Tensor *im_info_slice, const Tensor &anchors, const Tensor &variances,
    Tensor *bbox_deltas_slice, Tensor *scores_slice, int pre_nms_topN,
    int post_nms_topN, float nms_thresh, float min_size) {
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
  bbox_deltas_swap.mutable_data(place, bbox_deltas.type());
  scores_swap.mutable_data(place, scores.type());

  // Transpose bbox_deltas
  framework::AttributeMap trans_attrs;
  trans_attrs["axis"] = framework::vectorize2int({1, 2, 0});
  Transpose(scope, place, bbox_deltas, &bbox_deltas_swap, trans_attrs);

  // Transpose scores
  Transpose(scope, place, scores, &scores_swap, trans_attrs);

  scores_swap.Resize({scores_swap.numel(), 1});
  auto *scores_data = scores_swap.mutable_data<float>(place);

  // Sort index
  Tensor index_t;
  index_t.Resize({scores_swap.numel()});
  int *index = index_t.mutable_data<int>(place);
  for (int64_t i = 0; i < scores.numel(); ++i) {
    index[i] = i;
  }
  std::function<bool(const int64_t &, const int64_t &)> compare = [scores_data](
      const int64_t &i, const int64_t &j) {
    return scores_data[i] > scores_data[j];
  };
  if (pre_nms_topN <= 0 || pre_nms_topN >= scores_swap.numel()) {
    std::sort(index, index + scores.numel(), compare);
  } else {
    std::nth_element(index, index + pre_nms_topN, index + scores.numel(),
                     compare);
    index_t.Resize({pre_nms_topN});
  }

  Gather(scope, place, scores_swap, index_t, &scores);

  bbox_deltas_swap.Resize({bbox_deltas_swap.numel() / 4, 4});
  Gather(scope, place, bbox_deltas_swap, index_t, &bbox_deltas);

  Tensor all_anchors, all_anchors_swap;
  all_anchors_swap = anchors;
  all_anchors_swap.Resize({all_anchors_swap.numel() / 4, 4});
  all_anchors.Resize(all_anchors_swap.dims());
  all_anchors.mutable_data(place, all_anchors_swap.type());

  Gather(scope, place, all_anchors_swap, index_t, &all_anchors);

  Tensor proposals = BoxCoder(scope, place, &all_anchors, &bbox_deltas,
                              variances, "decode_center_size", false);

  ClipTiledBoxes(place, &proposals, im_info_slice);

  Tensor keep = FilterBoxes(place, &proposals, min_size, im_info_slice);

  Tensor proposals_swap;
  proposals_swap.Resize(proposals.dims());
  proposals_swap.mutable_data(place, proposals.type());

  Gather(scope, place, proposals, keep, &proposals_swap);
  Gather(scope, place, scores, keep, &scores_swap);

  std::cout << "proposals kept: " << std::endl;
  for (int64_t i = 0; i < proposals_swap.numel(); ++i) {
    std::cout << proposals_swap.data<float>()[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "scores kept: " << std::endl;
  for (int64_t i = 0; i < scores_swap.numel(); ++i) {
    std::cout << scores_swap.data<float>()[i] << " ";
  }
  std::cout << std::endl;

  if (nms_thresh <= 0) {
    return std::make_pair(proposals_swap, scores_swap);
  }
  /*
  NMS(scope, place, proposals_swap, scores_swap, keep);
  if (post_nms_topN > 0) {
    keep.Resize({post_nms_topN});
  }
  Gather(scope, place, proposals_swap, keep, proposals);
  Gather(scope, place, scores_swap, keep, scores);
  */
  return std::make_pair(proposals, scores);
}

class GenerateProposalsOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &scores = scope.FindVar(Input("Scores"))->Get<LoDTensor>();
    auto &bbox_deltas = scope.FindVar(Input("BboxDeltas"))->Get<LoDTensor>();
    auto &im_info = scope.FindVar(Input("ImInfo"))->Get<LoDTensor>();
    auto &anchors = scope.FindVar(Input("Anchors"))->Get<LoDTensor>();
    auto &variances = scope.FindVar(Input("Variances"))->Get<LoDTensor>();
    auto *rpn_rois = scope.FindVar(Output("RpnRois"))->GetMutable<LoDTensor>();
    auto *rpn_roi_probs =
        scope.FindVar(Output("RpnRoiProbs"))->GetMutable<LoDTensor>();

    // rpn_rois->mutable_data(place, anchors.type());
    // rpn_roi_probs->mutable_data(place, scores.type());

    rpn_rois->Resize(anchors.dims());
    rpn_roi_probs->Resize(scores.dims());
    framework::VisitDataType(framework::ToDataType(anchors.type()),
                             DyDataTypeVisitor(place, rpn_rois));
    framework::VisitDataType(framework::ToDataType(scores.type()),
                             DyDataTypeVisitor(place, rpn_roi_probs));

    int pre_nms_topN = Attr<int>("pre_nms_topN");
    int post_nms_topN = Attr<int>("post_nms_topN");
    float nms_thresh = Attr<float>("nms_thresh");
    float min_size = Attr<float>("min_size");

    paddle::framework::Scope &local_scope = scope.NewScope();

    framework::LoD lod;
    std::vector<size_t> lod0(1, 0);

    int64_t num_images = scores.dims()[0];
    int64_t num_proposals = 0;
    for (int64_t i = 0; i < num_images; ++i) {
      Tensor im_info_slice = im_info.Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas.Slice(i, i + 1);
      Tensor scores_slice = scores.Slice(i, i + 1);
      std::pair<Tensor, Tensor> tensor_pair = ProposalForOneImage(
          &local_scope, place, &im_info_slice, anchors, variances,
          &bbox_deltas_slice, &scores_slice, pre_nms_topN, post_nms_topN,
          nms_thresh, min_size);
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
                  ops::GenerateProposalsOpInferShape,
                  paddle::framework::EmptyGradOpMaker);
