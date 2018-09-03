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

#include <random>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

class RpnTargetAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("DistMat"),
                   "Input(DistMat) of RpnTargetAssignOp should not be null");

    PADDLE_ENFORCE(
        ctx->HasOutput("LocationIndex"),
        "Output(LocationIndex) of RpnTargetAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("ScoreIndex"),
        "Output(ScoreIndex) of RpnTargetAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("TargetLabel"),
        "Output(TargetLabel) of RpnTargetAssignOp should not be null");

    auto in_dims = ctx->GetInputDim("DistMat");
    PADDLE_ENFORCE_EQ(in_dims.size(), 2,
                      "The rank of Input(DistMat) must be 2.");

    ctx->SetOutputDim("LocationIndex", {-1});
    ctx->SetOutputDim("ScoreIndex", {-1});
    ctx->SetOutputDim("TargetLabel", {-1, 1});
    ctx->SetOutputDim("TargetBBox", {-1, 4});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<framework::LoDTensor>("DistMat")->type()),
        platform::CPUPlace());
  }
};

template <typename T>
class RpnTargetAssignKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* anchor_t = context.Input<Tensor>("Anchor");  // (H*W*A) * 4
    auto* gt_bbox_t = context.Input<Tensor>("GtBox");
    auto* dist_t = context.Input<LoDTensor>("DistMat");

    auto* loc_index_t = context.Output<Tensor>("LocationIndex");
    auto* score_index_t = context.Output<Tensor>("ScoreIndex");
    auto* tgt_bbox_t = context.Output<Tensor>("TargetBBox");
    auto* tgt_lbl_t = context.Output<Tensor>("TargetLabel");

    auto lod = dist_t->lod().back();
    int64_t batch_num = static_cast<int64_t>(lod.size() - 1);
    int64_t anchor_num = dist_t->dims()[1];
    PADDLE_ENFORCE_EQ(anchor_num, anchor_t->dims()[0]);

    int rpn_batch_size = context.Attr<int>("rpn_batch_size_per_im");
    float pos_threshold = context.Attr<float>("rpn_positive_overlap");
    float neg_threshold = context.Attr<float>("rpn_negative_overlap");
    float fg_fraction = context.Attr<float>("fg_fraction");

    int fg_num_per_batch = static_cast<int>(rpn_batch_size * fg_fraction);

    int64_t max_num = batch_num * anchor_num;
    auto place = context.GetPlace();

    tgt_bbox_t->mutable_data<T>({max_num, 4}, place);
    auto* loc_index = loc_index_t->mutable_data<int>({max_num}, place);
    auto* score_index = score_index_t->mutable_data<int>({max_num}, place);

    Tensor tmp_tgt_lbl;
    auto* tmp_lbl_data = tmp_tgt_lbl.mutable_data<int64_t>({max_num}, place);
    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, int64_t> iset;
    iset(dev_ctx, &tmp_tgt_lbl, static_cast<int64_t>(-1));

    std::random_device rnd;
    std::minstd_rand engine;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
    engine.seed(seed);

    int fg_num = 0;
    int bg_num = 0;
    for (int i = 0; i < batch_num; ++i) {
      Tensor dist = dist_t->Slice(lod[i], lod[i + 1]);
      Tensor gt_bbox = gt_bbox_t->Slice(lod[i], lod[i + 1]);
      auto fg_bg_gt = SampleFgBgGt(dev_ctx, dist, pos_threshold, neg_threshold,
                                   rpn_batch_size, fg_num_per_batch, engine,
                                   tmp_lbl_data + i * anchor_num);

      int cur_fg_num = fg_bg_gt[0].size();
      int cur_bg_num = fg_bg_gt[1].size();
      std::transform(fg_bg_gt[0].begin(), fg_bg_gt[0].end(), loc_index,
                     [i, anchor_num](int d) { return d + i * anchor_num; });
      memcpy(score_index, loc_index, cur_fg_num * sizeof(int));
      std::transform(fg_bg_gt[1].begin(), fg_bg_gt[1].end(),
                     score_index + cur_fg_num,
                     [i, anchor_num](int d) { return d + i * anchor_num; });

      // get target bbox deltas
      if (cur_fg_num) {
        Tensor fg_gt;
        T* gt_data = fg_gt.mutable_data<T>({cur_fg_num, 4}, place);
        Tensor tgt_bbox = tgt_bbox_t->Slice(fg_num, fg_num + cur_fg_num);
        T* tgt_data = tgt_bbox.data<T>();
        Gather<T>(anchor_t->data<T>(), 4,
                  reinterpret_cast<int*>(&fg_bg_gt[0][0]), cur_fg_num,
                  tgt_data);
        Gather<T>(gt_bbox.data<T>(), 4, reinterpret_cast<int*>(&fg_bg_gt[2][0]),
                  cur_fg_num, gt_data);
        BoxToDelta<T>(cur_fg_num, tgt_bbox, fg_gt, nullptr, false, &tgt_bbox);
      }

      loc_index += cur_fg_num;
      score_index += cur_fg_num + cur_bg_num;
      fg_num += cur_fg_num;
      bg_num += cur_bg_num;
    }

    int lbl_num = fg_num + bg_num;
    PADDLE_ENFORCE_LE(fg_num, max_num);
    PADDLE_ENFORCE_LE(lbl_num, max_num);

    tgt_bbox_t->Resize({fg_num, 4});
    loc_index_t->Resize({fg_num});
    score_index_t->Resize({lbl_num});
    auto* lbl_data = tgt_lbl_t->mutable_data<int64_t>({lbl_num, 1}, place);
    Gather<int64_t>(tmp_lbl_data, 1, score_index_t->data<int>(), lbl_num,
                    lbl_data);
  }

 private:
  void ScoreAssign(const T* dist_data, const Tensor& anchor_to_gt_max,
                   const int row, const int col, const float pos_threshold,
                   const float neg_threshold, int64_t* target_label,
                   std::vector<int>* fg_inds, std::vector<int>* bg_inds) const {
    float epsilon = 0.0001;
    for (int64_t i = 0; i < row; ++i) {
      const T* v = dist_data + i * col;
      T max = *std::max_element(v, v + col);
      for (int64_t j = 0; j < col; ++j) {
        if (std::abs(max - v[j]) < epsilon) {
          target_label[j] = 1;
        }
      }
    }

    // Pick the fg/bg
    const T* anchor_to_gt_max_data = anchor_to_gt_max.data<T>();
    for (int64_t j = 0; j < col; ++j) {
      if (anchor_to_gt_max_data[j] >= pos_threshold) {
        target_label[j] = 1;
      } else if (anchor_to_gt_max_data[j] < neg_threshold) {
        target_label[j] = 0;
      }
      if (target_label[j] == 1) {
        fg_inds->push_back(j);
      } else if (target_label[j] == 0) {
        bg_inds->push_back(j);
      }
    }
  }

  void ReservoirSampling(const int num, std::minstd_rand engine,
                         std::vector<int>* inds) const {
    std::uniform_real_distribution<float> uniform(0, 1);
    size_t len = inds->size();
    if (len > static_cast<size_t>(num)) {
      for (size_t i = num; i < len; ++i) {
        int rng_ind = std::floor(uniform(engine) * i);
        if (rng_ind < num)
          std::iter_swap(inds->begin() + rng_ind, inds->begin() + i);
      }
      inds->resize(num);
    }
  }

  // std::vector<std::vector<int>> RpnTargetAssign(
  std::vector<std::vector<int>> SampleFgBgGt(
      const platform::CPUDeviceContext& ctx, const Tensor& dist,
      const float pos_threshold, const float neg_threshold,
      const int rpn_batch_size, const int fg_num, std::minstd_rand engine,
      int64_t* target_label) const {
    auto* dist_data = dist.data<T>();
    int row = dist.dims()[0];
    int col = dist.dims()[1];

    std::vector<int> fg_inds;
    std::vector<int> bg_inds;
    std::vector<int> gt_inds;

    // Calculate the max IoU between anchors and gt boxes
    // Map from anchor to gt box that has highest overlap
    auto place = ctx.GetPlace();
    Tensor anchor_to_gt_max, anchor_to_gt_argmax;
    anchor_to_gt_max.mutable_data<T>({col}, place);
    int* argmax = anchor_to_gt_argmax.mutable_data<int>({col}, place);

    auto x = framework::EigenMatrix<T>::From(dist);
    auto x_col_max = framework::EigenVector<T>::Flatten(anchor_to_gt_max);
    auto x_col_argmax =
        framework::EigenVector<int>::Flatten(anchor_to_gt_argmax);
    x_col_max = x.maximum(Eigen::DSizes<int, 1>(0));
    x_col_argmax = x.argmax(0).template cast<int>();

    // Follow the Faster RCNN's implementation
    ScoreAssign(dist_data, anchor_to_gt_max, row, col, pos_threshold,
                neg_threshold, target_label, &fg_inds, &bg_inds);
    // Reservoir Sampling
    ReservoirSampling(fg_num, engine, &fg_inds);
    int fg_num2 = static_cast<int>(fg_inds.size());
    int bg_num = rpn_batch_size - fg_num2;
    ReservoirSampling(bg_num, engine, &bg_inds);

    gt_inds.reserve(fg_num2);
    for (int i = 0; i < fg_num2; ++i) {
      gt_inds.emplace_back(argmax[fg_inds[i]]);
    }
    std::vector<std::vector<int>> fg_bg_gt;
    fg_bg_gt.emplace_back(fg_inds);
    fg_bg_gt.emplace_back(bg_inds);
    fg_bg_gt.emplace_back(gt_inds);

    return fg_bg_gt;
  }
};

class RpnTargetAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Anchor",
             "(Tensor) input anchor is a 2-D Tensor with shape [H*W*A, 4].");
    AddInput("GtBox", "(LoDTensor) input groud-truth bbox with shape [K, 4].");
    AddInput(
        "DistMat",
        "(LoDTensor or Tensor) this input is a 2-D LoDTensor with shape "
        "[K, M]. It is pair-wise distance matrix between the entities "
        "represented by each row and each column. For example, assumed one "
        "entity is A with shape [K], another entity is B with shape [M]. The "
        "DistMat[i][j] is the distance between A[i] and B[j]. The bigger "
        "the distance is, the better macthing the pairs are. Please note, "
        "This tensor can contain LoD information to represent a batch of "
        "inputs. One instance of this batch can contain different numbers of "
        "entities.");
    AddAttr<float>(
        "rpn_positive_overlap",
        "Minimum overlap required between an anchor and ground-truth "
        "box for the (anchor, gt box) pair to be a positive example.")
        .SetDefault(0.7);
    AddAttr<float>(
        "rpn_negative_overlap",
        "Maximum overlap allowed between an anchor and ground-truth "
        "box for the (anchor, gt box) pair to be a negative examples.")
        .SetDefault(0.3);
    AddAttr<float>(
        "fg_fraction",
        "Target fraction of RoI minibatch that "
        "is labeled foreground (i.e. class > 0), 0-th class is background.")
        .SetDefault(0.25);
    AddAttr<int>("rpn_batch_size_per_im",
                 "Total number of RPN examples per image.")
        .SetDefault(256);
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest.")
        .SetDefault(false);
    AddAttr<int>("seed", "RpnTargetAssign random seed.").SetDefault(0);
    AddOutput(
        "LocationIndex",
        "(Tensor), The indexes of foreground anchors in all RPN anchors, the "
        "shape of the LocationIndex is [F], F depends on the value of input "
        "tensor and attributes.");
    AddOutput(
        "ScoreIndex",
        "(Tensor), The indexes of foreground and background anchors in all "
        "RPN anchors(The rest anchors are ignored). The shape of the "
        "ScoreIndex is [F + B], F and B are sampled foreground and backgroud "
        " number.");
    AddOutput("TargetBBox",
              "(Tensor<int64_t>), The target bbox deltas with shape "
              "[F, 4], F is the sampled foreground number.");
    AddOutput(
        "TargetLabel",
        "(Tensor<int64_t>), The target labels of each anchor with shape "
        "[F + B, 1], F and B are sampled foreground and backgroud number.");
    AddComment(R"DOC(
This operator can be, for given the IoU between the ground truth bboxes and the
anchors, to assign classification and regression targets to each prediction.
The Score index and LocationIndex will be generated according to the DistMat.
The rest anchors would not contibute to the RPN training loss

ScoreIndex is composed of foreground anchor indexes(positive labels) and
background anchor indexes(negative labels). LocationIndex is exactly same
as the foreground anchor indexes since we can not assign regression target to 
the background anchors.

The classification targets(TargetLabel) is a binary class label (of being
an object or not). Following the paper of Faster-RCNN, the positive labels
are two kinds of anchors: (i) the anchor/anchors with the highest IoU
overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap
higher than rpn_positive_overlap(0.7) with any ground-truth box. Note that
a single ground-truth box may assign positive labels to multiple anchors.
A non-positive anchor is when its IoU ratio is lower than rpn_negative_overlap
(0.3) for all ground-truth boxes. Anchors that are neither positive nor
negative do not contribute to the training objective.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(rpn_target_assign, ops::RpnTargetAssignOp,
                  ops::RpnTargetAssignOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(rpn_target_assign, ops::RpnTargetAssignKernel<float>,
                       ops::RpnTargetAssignKernel<double>);
