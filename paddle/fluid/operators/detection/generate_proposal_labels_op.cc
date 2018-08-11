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
    auto im_info_dims = ctx->GetInputDim("ImInfo");

    // TODO(BUXINGYUAN)
    // INPUT DIM CHECK
  }
};

void ConcatAxis0(framework::Scope *scope, const platform::Place &place,
                 const Tensor &in_tenosr_a, const Tensor &in_tensor_b,
                 Tensor *out_tensor) {
  std::string in_name_a, in_name_b, out_name;
  scope->Var(&in_name_a)->GetMutable<LoDTensor>()->ShareDataWith(in_tensor_a);
  scope->Var(&in_name_b)->GetMutable<LoDTensor>()->ShareDataWith(in_tensor_b);
  scope->Var(&out_name)->GetMutable<LoDTensor>()->ShareDataWith(out_tensor);

  framework::AttributeMap attrs;
  attrs["axis"] = 0;
  auto concat_op = framework::OpRegistry::CreateOp(
      "concat", {{"X", {in_name_a, in_name_b}}}, {{"Out", {out_name}}}, attrs);
  concat_op->Run(*scope, place);
  out_tensor->Resize(scope->FindVar(out_name)->GetMutable<LoDTensor>()->dims());
}

void ScaleMul(framework::Scope *scope, const platform::Place &place,
              const Tensor &in_tensor, Tensor *out_tensor,
              const framework::AttributeMap &attrs) {
  std::string in_name, out_name;
  scope->Var(&in_name)->GetMutable<LoDTensor>()->ShareDataWith(in_tensor);
  scope->Var(&out_name)->GetMutable<LoDTensor>()->ShareDataWith(out_tensor);

  auto scalemul_op = framework::OpRegistry::CreateOp(
      "scale", {{"X", {in_name}}}, {{"Out", {out_name}}}, attrs);
  scalemul_op->Run(*scope, place);

  out_tensor->Resize(scope->FindVar(out_name)->GetMutable<LoDTensor>()->dims());
}

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

std::vector<Tensor> SampleRoisForOneImage(
    framework::Scope *scope, const paltform::Place &place,
    Tensor *rpn_rois_slice, Tensor *gt_classes_slice, Tensor *gt_boxes_slice,
    float *im_scale, int batch_size_per_im, float fg_fraction, float fg_thresh,
    float bg_thresh_hi, float bg_thresh_lo, std::vector<float> bbox_reg_weights,
    int class_nums) {
  Tensor rpn_rois, rpn_rois_swap;
  Tensor gt_classes, gt_classes_swap;
  Tensor gt_boxes, gt_boxes_swap;
  Tensor im_scalse, im_scalse_swap;

  framework::DDim rpn_rois_shape = framework::slice_ddim(
      rpn_rois_slice->dims(), 1, rpn_rois_slice->dims().size());
  framework::DDim gt_classes_shape = framework::slice_ddim(
      gt_classes_slice->dims(), 1, gt_classes_slice->dims().size());
  framework::DDim gt_boxes_shape = framework::slice_ddim(
      gt_boxes_slice->dims(), 1, gt_boxes_slice->dims().size());

  rpn_rois = *rpn_rois_slice;
  gt_classes = *gt_classes_slice;
  gt_boxes = *gt_boxes_slice;
  rpn_rois.Resize(rpn_rois_shape);
  gt_classes.Resize(gt_classes_shape);
  gt_boxes.Resize(gt_boxes_shape);

  bbox_deltas_swap.Resize(bbox_deltas_shape);
  scores_swap.Resize(scores_shape);
  bbox_deltas_swap.mutable_data(place, bbox_deltas.type());
  scores_swap.mutable_data(place, scores.type());

  // Roidb
  framework::AttributeMap scale_attrs;
  scale_attrs["scale"] = 1 / im_scale;
  ScaleMul(scope, place, rpn_rois, &rpn_rois_swap, scale_attrs);
  Tensor boxes;
  ConcatAxis0(scope, place, gt_boxes, rpn_rois, &boxes);

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

  return std::make_pair(proposals, scores);
}

class GenerateProposalLabelsOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &rpn_rois = scope.FindVar(Input("RpnRois"))->Get<LoDTensor>();
    auto &gt_classes = scope.FindVar(Input("GtClasses"))->Get<LoDTensor>();
    auto &gt_boxes = scope.FindVar(Input("GtBoxes"))->Get<LoDTensor>();
    auto &im_scales = scope.FindVar(Input("ImScales"))->Get<LoDTensor>();

    auto *rois = scope.FindVar(Output("Rois"))->GetMutable<LoDTensor>();
    auto *label_int32 =
        scope.FindVar(Output("LabelsInt32"))->GetMutable<LoDTensor>();
    auto *bbox_targets =
        scope.FindVar(Output("BboxTargets"))->GetMutable<LoDTensor>();
    auto *bbox_inside_weights =
        scope.FindVar(Output("BboxInsideWeights"))->GetMutable<LoDTensor>();
    auto *bbox_outside_weights =
        scope.FindVar(Output("BboxOutsideWeights"))->GetMutable<LoDTensor>();
    rois->Resize(rpn_rois.dims());
    label_int32->Resize(rpn_rois.dims());
    bbox_targets->Resize(rpn_rois.dims());
    bbox_inside_weights->Resize(rpn_rois.dims());
    bbox_outside_weights->Resize(rpn_rois.dims());
    framework::VisitDataType(framework::ToDataType(rpn_rois.type()),
                             DyDataTypeVisitor(place, rois));
    framework::VisitDataType(framework::ToDataType(gt_classes.type()),
                             DyDataTypeVisitor(place, label_int32));
    framework::VisitDataType(framework::ToDataType(rpn_rois.type()),
                             DyDataTypeVisitor(place, bbox_targets));
    framework::VisitDataType(framework::ToDataType(rpn_rois.type()),
                             DyDataTypeVisitor(place, bbox_inside_weights));
    framework::VisitDataType(framework::ToDataType(rpn_rois.type()),
                             DyDataTypeVisitor(place, bbox_outside_weights));

    int batch_size_per_im = Attr<int>("batch_size_per_im");
    float fg_fraction = Attr<int>("fg_fraction");
    float fg_thresh = Attr<float>("fg_thresh");
    float bg_thresh_hi = Attr<float>("bg_thresh_hi");
    float bg_thresh_lo = Attr<float>("bg_thresh_lo");
    std::vector<float> bbox_reg_weights =
        Attr<std::vector<float>>("bbox_reg_weights");
    int class_nums = Attr<float>("class_nums");

    paddle::framework::Scope &local_scope = scope.NewScope();

    framework::LoD lod;
    std::vector<size_t> lod0(1, 0);

    int64_t num_images = im_scales.dims()[0];
    int64_t num_rois = 0;
    for (int64_t i = 0; i < num_images; ++i) {
      Tensor rpn_rois_slice = rpn_rois.Slice(i, i + 1);
      Tensor gt_classes_slice = gt_classes.Slice(i, i + 1);
      Tensor gt_boxes_slice = gt_boxes.Slice(i, i + 1);
      Tensor im_scales_slice = im_scalse.Slice(i, i + 1);
      std::vector<Tensor> tensor_output = SampleRoisForOneImage(
          &local_scope, place, &rpn_rois_slice, &gt_classes_slice,
          &gt_boxes_slice, &im_scalse_slice, batch_size_per_im, fg_fraction,
          fg_thresh, bg_thresh_hi, bg_thresh_lo, bbox_reg_weights, class_nums);
      sampled_rois = tensor_output[0];
      sampled_label_int32 = tensor_output[1];
      sampled_bbox_targets = tensor_output[2];
      sampled_bbox_inside_weights = tensor_output[3];
      sampled_bbox_outside_weights = tensor_output[4];

      framework::VisitDataType(
          framework::ToDataType(rois->type()),
          AppendRoisFunctor(rois, 4 * num_rois, &sampled_rois));
      framework::VisitDataType(
          framework::ToDataType(label_int32->type()),
          AppendRoisFunctor(label_int32, num_rois, &sampled_label_int32));
      framework::VisitDataType(
          framework::ToDataType(bbox_targets->type()),
          AppendRoisFunctor(bbox_targets, 4 * num_rois, &sampled_bbox_targets));
      framework::VisitDataType(
          framework::ToDataType(bbox_inside_weights->type()),
          AppendRoisFunctor(bbox_inside_weights, 4 * num_rois,
                            &sampled_bbox_inside_weights));
      framework::VisitDataType(
          framework::ToDataType(bbox_outside_weights->type()),
          AppendRoisFunctor(bbox_outside_weights, 4 * num_rois,
                            &sampled_bbox_outside_weights));

      num_rois += sampled_rois.dims()[0];
      lod0.emplace_back(num_rois);
    }

    lod.emplace_back(lod0);
    rois->set_lod(lod);
    label_int32->set_lod(lod);
    bbox_targets->set_lod(lod);
    bbox_inside_weights->set_lod(lod);
    bbox_outside_weights->set_lod(lod);
    rois->Resize({num_rois, 4});
    label_int32->Resize({num_rois, 1});
    bbox_targets->Resize({num_rois, 4});
    bbox_inside_weights->Resize({num_rois, 4});
    bbox_outside_weights->Resize({num_rois, 4});
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
