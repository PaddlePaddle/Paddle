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

    std::cout << "in infershape " << std::endl;
    ctx->SetOutputDim("RpnRois", scores_dims);
    ctx->SetOutputDim("RpnRoiProbs", scores_dims);
  }
};

void Transpose(framework::Scope &scope, const platform::Place &place,
               std::string &in_name, Tensor &in_tensor, std::string &out_name,
               Tensor &out_tensor, framework::AttributeMap &attrs) {
  scope.Var(&in_name)->GetMutable<LoDTensor>()->ShareDataWith(in_tensor);
  scope.Var(&out_name)->GetMutable<LoDTensor>()->ShareDataWith(out_tensor);
  auto transpose_op = framework::OpRegistry::CreateOp(
      "transpose", {{"X", {in_name}}}, {{"Out", {out_name}}}, attrs);
  transpose_op->Run(scope, place);
}

void Gather(framework::Scope &scope, const platform::Place &place,
            std::string &in_name, std::string &index_name,
            std::string &out_name) {
  framework::AttributeMap attrs;
  auto gather_op = framework::OpRegistry::CreateOp(
      "gather", {{"X", {in_name}}, {"Index", {index_name}}},
      {{"Out", {out_name}}}, attrs);
  gather_op->Run(scope, place);
}

void BoxCoder(framework::Scope &scope, const platform::Place &place,
              Tensor &all_anchors, Tensor &bbox_deltas, Tensor &variances,
              Tensor &proposals, std::string code_type, bool box_normalized) {
  std::vector<int64_t> bbox_deltas_shape =
      framework::vectorize(bbox_deltas.dims());
  bbox_deltas_shape.emplace(bbox_deltas_shape.begin(), 1);
  bbox_deltas.Resize(framework::make_ddim(bbox_deltas_shape));

  proposals.Resize(all_anchors.dims());
  proposals.mutable_data(place, all_anchors.type());

  std::string all_anchors_name, bbox_deltas_name, variances_name,
      proposals_name;
  framework::AttributeMap attrs;
  attrs["code_type"] = code_type;
  attrs["box_normalized"] = box_normalized;
  scope.Var(&all_anchors_name)
      ->GetMutable<LoDTensor>()
      ->ShareDataWith(all_anchors);
  scope.Var(&bbox_deltas_name)
      ->GetMutable<Tensor>()
      ->ShareDataWith(bbox_deltas);
  scope.Var(&variances_name)->GetMutable<Tensor>()->ShareDataWith(variances);
  scope.Var(&proposals_name)->GetMutable<Tensor>()->ShareDataWith(bbox_deltas);
  auto box_coder_op = framework::OpRegistry::CreateOp(
      "box_coder", {{"PriorBox", {all_anchors_name}},
                    {"PriorBoxVar", {invariances_name}},
                    {"TargetBox", {bbox_deltas_name}},
                    {{"OutputBox", proposals_name}}},
      attrs);
  box_coder_op->Run(scope, place);

  bbox_deltas_shape =
      framework::slice_ddim(bbox_deltas.dims(), 1, bbox_deltas.dims().size());
  framework::DDim proposals_shape =
      framework::slice_ddim(proposals.dims(), 1, proposals.dims().size());
  bbox_deltas.Resize(bbox_deltas_shape);
  proposals.Resize(proposals_shape);
}

void ClipTiledBoxes(const platform::Place &place, Tensor &boxes,
                    const Tensor &im_info) {
  float *boxes_data = boxes.data<float>(place);
  float *im_info_data = im_info.data<float>(place);
  for (int64_t i = 0; i < boxes.numel(); ++i) {
    if (i % 4 == 0) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), 0.0);
    } else if (i % 4 == 1) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[0] - 1), 0.0);
    } else if (i % 4 == 2) {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[1] - 1), 0.0);
    } else {
      boxes_data[i] =
          std::max(std::min(boxes_data[i], im_info_data[0] - 1), 0.0);
    }
  }
}

Tensor FilterBoxes(const platform::Place &place, Tensor &proposals,
                   float min_size, Tensor &im_info) {
  float *im_info_data = im_info.data<float>(place);
  float *boxes_data = proposals.data<float>(place);
  min_size *= im_info_data[2];
  Tensor keep;
  keep.Resize({1, proposals.dims()[0]});
  int64_t *keep_data = keep.data<float>(place);

  int64_t keep_len = 0;
  for (int64_t i = 0; i < keep.numel(); ++i) {
    float ws = proposals[4 * i + 2] - proposals[4 * i] + 1;
    float hs = proposals[4 * i + 3] - proposals[4 * i + 1] + 1;
    float x_ctr = proposals[4 * i] + ws / 2;
    float y_ctr = proposals[4 * i + 1] + hs / 2;
    if (ws >= min_size && hs >= min_size && x_ctr <= im_info[1] &&
        y_ctr <= im_info[0]) {
      keep_data[keep_len++] = i;
    }
  }
  keep.Resize({1, keep_len});
  return keep;
}

Tensor NMS(const platform::Place &place, Tensor &proposals, float min_size,
           Tensor &im_info) {
  float *im_info_data = im_info.data<float>(place);
  float *boxes_data = proposals.data<float>(place);
  min_size *= im_info_data[2];
  Tensor keep;
  keep.Resize({1, proposals.dims()[0]});
  int64_t *keep_data = keep.data<float>(place);

  int64_t keep_len = 0;
  for (int64_t i = 0; i < keep.numel(); ++i) {
    float ws = proposals[4 * i + 2] - proposals[4 * i] + 1;
    float hs = proposals[4 * i + 3] - proposals[4 * i + 1] + 1;
    float x_ctr = proposals[4 * i] + ws / 2;
    float y_ctr = proposals[4 * i + 1] + hs / 2;
    if (ws >= min_size && hs >= min_size && x_ctr <= im_info[1] &&
        y_ctr <= im_info[0]) {
      keep_data[keep_len++] = i;
    }
  }
  keep.Resize({1, keep_len});
  return keep;
}

//<typename T>
void ProposalForOneImage(framework::Scope &scope, const platform::Place &place,
                         const Tensor &im_info, const Tensor &anchors,
                         const Tensor &variances, Tensor &bbox_deltas,
                         Tensor &scores, int pre_nms_topN, int post_nms_topN,
                         float nms_thresh, float min_size) {
  Tensor bbox_deltas_trans;
  Tensor scores_trans;

  framework::DDim bbox_deltas_shape =
      framework::slice_ddim(bbox_deltas.dims(), 1, bbox_deltas.dims().size());
  framework::DDim scores_shape =
      framework::slice_ddim(scores.dims(), 1, scores.dims().size());

  bbox_deltas.Resize(bbox_deltas_shape);
  scores.Resize(scores_shape);
  bbox_deltas_trans.Resize(bbox_deltas_shape);
  scores_trans.Resize(scores_shape);

  bbox_deltas_trans.mutable_data(place, bbox_deltas.type());
  scores_trans.mutable_data(place, scores.type());
  std::cout << "in one image" << std::endl;

  // Transpose bbox_deltas
  framework::AttributeMap trans_attrs;
  trans_attrs["axis"] = framework::vectorize2int({1, 2, 0});
  std::string bbox_deltas_var_name, bbox_deltas_trans_var_name;
  Transpose(scope, place, bbox_deltas_var_name, bbox_deltas,
            bbox_deltas_trans_var_name, bbox_deltas_trans, trans_attrs);

  // Transpose scores
  std::string scores_var_name, scores_trans_var_name;
  Transpose(scope, place, scores_var_name, scores, scores_trans_var_name,
            scores_trans, trans_attrs);

  auto *scores_data = scores_trans.mutable_data<float>(place);

  // Sort index
  Tensor index_t;
  index_t.Resize({1, scores_trans.numel()});
  int64_t *index = index_t.mutable_data<int64_t>(place);
  for (int64_t i = 0; i < scores.numel(); ++i) {
    index[i] = i;
  }
  std::function<bool(const int64_t &, const int64_t &)> compare = [scores_data](
      const int64_t &i, const int64_t &j) {
    return scores_data[i] > scores_data[j];
  };
  if (pre_nms_topN <= 0 || pre_nms_topN >= scores_trans.numel()) {
    std::sort(index, index + scores.numel(), compare);
  } else {
    std::nth_element(index, index + pre_nms_topN, index + scores.numel(),
                     compare);
    index_t.Resize({1, pre_nms_topN});
  }

  for (int i = 0; i < index_t.numel(); ++i) {
    std::cout << index[i] << " ";
  }
  std::cout << "\n";

  // Gather
  std::string index_name;
  scope.Var(&index_name)->GetMutable<LoDTensor>()->ShareDataWith(index_t);
  Gather(scope, place, scores_trans_var_name, index_name, scores_var_name);
  Gather(scope, place, bbox_deltas_trans_var_name, index_name,
         bbox_deltas_var_name);

  std::string anchors_var_name, all_anchors_var_name;
  Tensor all_anchors;
  all_anchors.Resize(anchors.dims());
  all_anchors.mutable_data(place, anchors.type());
  scope.Var(&anchors_var_name)->GetMutable<Tensor>()->ShareDataWith(anchors);
  scope.Var(&all_anchors_var_name)
      ->GetMutable<Tensor>()
      ->ShareDataWith(all_anchors);
  Gather(scope, place, anchors_var_name, index_name, all_anchors_var_name);

  // box_coder();
  Tensor proposals;
  BoxCoder(scope, place, all_anchors, bbox_deltas, variances, proposals,
           "DecodeCenterSize", false);

  ClipTiledBoxes(place, proposals, im_info);

  Tensor keep = FilterBoxes(place, proposals, min_size, im_info);

  Tensor proposals_keep, scores_keep;
  std::string keep_name, proposals_name, proposals_keep_name, scores_keep_name;
  proposals_keep.mutable_data(place, proposals.type());
  scores_keep.mutable_data(place, scores.type());
  scope.Var(&keep_name)->GetMutable<Tensor>()->ShareDataWith(keep);
  scope.Var(&proposals_name)->GetMutable<Tensor>()->ShareDataWith(proposals);
  scope.Var(&proposals_keep_name)
      ->GetMutable<Tensor>()
      ->ShareDataWith(proposals_keep);
  scope.Var(&scores_keep_name)
      ->GetMutable<Tensor>()
      ->ShareDataWith(scores_keep);

  Gather(scope, place, proposals_name, keep_name, proposals_keep_name);
  Gather(scope, place, scores_name, keep_name, scores_keep_name);

  if (nms_thresh <= 0) {
    return proposals_keep, scores_keep;
  }
}

class GenerateProposalsOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    std::cout << " this is a two test" << std::endl;
    auto &scores = scope.FindVar(Input("Scores"))->Get<LoDTensor>();
    std::cout << " this is a two test" << std::endl;
    auto &bbox_deltas = scope.FindVar(Input("BboxDeltas"))->Get<LoDTensor>();
    std::cout << " this is a two test" << std::endl;
    auto &im_info = scope.FindVar(Input("ImInfo"))->Get<LoDTensor>();
    std::cout << " this is a two test" << std::endl;
    auto &anchors = scope.FindVar(Input("Anchors"))->Get<LoDTensor>();
    std::cout << " this is a two test" << std::endl;
    auto &variances = scope.FindVar(Input("Variances"))->Get<LoDTensor>();
    std::cout << "in kernel begin" << std::endl;
    auto *rpn_rois = scope.FindVar(Output("RpnRois"))->GetMutable<LoDTensor>();
    auto *rpn_roi_probs =
        scope.FindVar(Output("RpnRoiProbs"))->GetMutable<LoDTensor>();

    rpn_rois->mutable_data(place, anchors.type());
    rpn_roi_probs->mutable_data(place, anchors.type());

    int pre_nms_topN = Attr<int>("pre_nms_topN");
    int post_nms_topN = Attr<int>("post_nms_topN");
    float nms_thresh = Attr<float>("nms_thresh");
    float min_size = Attr<float>("min_size");

    std::cout << "before for loop" << std::endl;

    paddle::framework::Scope &local_scope = scope.NewScope();

    int64_t num_images = scores.dims()[0];
    for (int64_t i = 0; i < num_images; ++i) {
      Tensor im_info_i = im_info.Slice(i, i + 1);
      Tensor bbox_deltas_i = bbox_deltas.Slice(i, i + 1);
      Tensor scores_i = scores.Slice(i, i + 1);
      ProposalForOneImage(local_scope, place, im_info_i, anchors, variances,
                          bbox_deltas_i, scores_i, pre_nms_topN, post_nms_topN,
                          nms_thresh, min_size);
    }
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
