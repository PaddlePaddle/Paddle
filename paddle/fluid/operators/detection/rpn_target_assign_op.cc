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
  }
};

template <typename T>
class RpnTargetAssignKernel : public framework::OpKernel<T> {
 public:
  void ScoreAssign(const T* dist_data, const Tensor& anchor_to_gt_max,
                   const int row, const int col, const float pos_threshold,
                   const float neg_threshold, int64_t* target_label_data,
                   std::vector<int>* fg_inds, std::vector<int>* bg_inds) const {
    int fg_offset = fg_inds->size();
    int bg_offset = bg_inds->size();
    for (int64_t i = 0; i < row; ++i) {
      const T* v = dist_data + i * col;
      T max_dist = *std::max_element(v, v + col);
      for (int64_t j = 0; j < col; ++j) {
        T val = dist_data[i * col + j];
        if (val == max_dist) target_label_data[j] = 1;
      }
    }

    // Pick the fg/bg and count the number
    for (int64_t j = 0; j < col; ++j) {
      if (anchor_to_gt_max.data<T>()[j] > pos_threshold) {
        target_label_data[j] = 1;
      } else if (anchor_to_gt_max.data<T>()[j] < neg_threshold) {
        target_label_data[j] = 0;
      }
      if (target_label_data[j] == 1) {
        fg_inds->push_back(fg_offset + j);
      } else if (target_label_data[j] == 0) {
        bg_inds->push_back(bg_offset + j);
      }
    }
  }

  void ReservoirSampling(const int num, const int offset,
                         std::minstd_rand engine,
                         std::vector<int>* inds) const {
    std::uniform_real_distribution<float> uniform(0, 1);
    const int64_t size = static_cast<int64_t>(inds->size());
    if (size > num) {
      for (int64_t i = num; i < size; ++i) {
        int rng_ind = std::floor(uniform(engine) * i);
        if (rng_ind < num)
          std::iter_swap(inds->begin() + rng_ind + offset,
                         inds->begin() + i + offset);
      }
    }
  }

  void RpnTargetAssign(const framework::ExecutionContext& ctx,
                       const Tensor& dist, const float pos_threshold,
                       const float neg_threshold, const int rpn_batch_size,
                       const int fg_num, std::minstd_rand engine,
                       std::vector<int>* fg_inds, std::vector<int>* bg_inds,
                       int64_t* target_label_data) const {
    auto* dist_data = dist.data<T>();
    int64_t row = dist.dims()[0];
    int64_t col = dist.dims()[1];
    int fg_offset = fg_inds->size();
    int bg_offset = bg_inds->size();

    // Calculate the max IoU between anchors and gt boxes
    Tensor anchor_to_gt_max;
    anchor_to_gt_max.mutable_data<T>(
        framework::make_ddim({static_cast<int64_t>(col), 1}),
        platform::CPUPlace());
    auto& place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    auto x = EigenMatrix<T>::From(dist);
    auto x_col_max = EigenMatrix<T>::From(anchor_to_gt_max);
    x_col_max.device(place) =
        x.maximum(Eigen::DSizes<int, 1>(0))
            .reshape(Eigen::DSizes<int, 2>(static_cast<int64_t>(col), 1));
    // Follow the Faster RCNN's implementation
    ScoreAssign(dist_data, anchor_to_gt_max, row, col, pos_threshold,
                neg_threshold, target_label_data, fg_inds, bg_inds);
    // Reservoir Sampling
    ReservoirSampling(fg_num, fg_offset, engine, fg_inds);
    int bg_num = rpn_batch_size - fg_inds->size();
    ReservoirSampling(bg_num, bg_offset, engine, bg_inds);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* dist = context.Input<LoDTensor>("DistMat");
    auto* loc_index = context.Output<Tensor>("LocationIndex");
    auto* score_index = context.Output<Tensor>("ScoreIndex");
    auto* tgt_lbl = context.Output<Tensor>("TargetLabel");

    auto col = dist->dims()[1];
    int64_t n = dist->lod().size() == 0UL
                    ? 1
                    : static_cast<int64_t>(dist->lod().back().size() - 1);
    if (dist->lod().size()) {
      PADDLE_ENFORCE_EQ(dist->lod().size(), 1UL,
                        "Only support 1 level of LoD.");
    }
    int rpn_batch_size = context.Attr<int>("rpn_batch_size_per_im");
    float pos_threshold = context.Attr<float>("rpn_positive_overlap");
    float neg_threshold = context.Attr<float>("rpn_negative_overlap");
    float fg_fraction = context.Attr<float>("fg_fraction");

    int fg_num = static_cast<int>(rpn_batch_size * fg_fraction);

    int64_t* target_label_data =
        tgt_lbl->mutable_data<int64_t>({n * col, 1}, context.GetPlace());

    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, int64_t> iset;
    iset(dev_ctx, tgt_lbl, static_cast<int>(-1));

    std::vector<int> fg_inds;
    std::vector<int> bg_inds;
    std::random_device rnd;
    std::minstd_rand engine;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
    engine.seed(seed);

    if (n == 1) {
      RpnTargetAssign(context, *dist, pos_threshold, neg_threshold,
                      rpn_batch_size, fg_num, engine, &fg_inds, &bg_inds,
                      target_label_data);
    } else {
      auto lod = dist->lod().back();
      for (size_t i = 0; i < lod.size() - 1; ++i) {
        Tensor one_ins = dist->Slice(lod[i], lod[i + 1]);
        RpnTargetAssign(context, one_ins, pos_threshold, neg_threshold,
                        rpn_batch_size, fg_num, engine, &fg_inds, &bg_inds,
                        target_label_data + i * col);
      }
    }
    int* loc_index_data = loc_index->mutable_data<int>(
        {static_cast<int>(fg_inds.size())}, context.GetPlace());
    int* score_index_data = score_index->mutable_data<int>(
        {static_cast<int>(fg_inds.size() + bg_inds.size())},
        context.GetPlace());
    memcpy(loc_index_data, reinterpret_cast<int*>(&fg_inds[0]),
           fg_inds.size() * sizeof(int));
    memcpy(score_index_data, reinterpret_cast<int*>(&fg_inds[0]),
           fg_inds.size() * sizeof(int));
    memcpy(score_index_data + fg_inds.size(),
           reinterpret_cast<int*>(&bg_inds[0]), bg_inds.size() * sizeof(int));
  }
};

class RpnTargetAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
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
        "ScoreIndex is [F + B], F and B depend on the value of input "
        "tensor and attributes.");
    AddOutput("TargetLabel",
              "(Tensor<int64_t>), The target labels of each anchor with shape "
              "[K * M, 1], "
              "K and M is the same as they are in DistMat.");
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
