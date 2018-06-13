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

class RpnTargetAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}
};

template <typename T>
class RpnTargetAssignKernel : public framework::OpKernel<T> {
 public:
  void AnchorToGt(const T* dist_data, const int64_t row, const int64_t col,
                  std::vector<float>* anchor_to_gt_max) const {
    for (int64_t j = 0; j < col; ++j) {
      T max_dist = -1;
      for (int64_t i = 0; i < row; ++i) {
        T val = dist_data[i * col + j];
        if (val > max_dist) max_dist = val;
      }
      anchor_to_gt_max->push_back(max_dist);
    }
  }

  void ScoreAssign(const T* dist_data,
                   const std::vector<float> anchor_to_gt_max, const int row,
                   const int col, const float pos_threshold,
                   const float neg_threshold, int* target_label_data,
                   std::vector<int>* fg_inds, std::vector<int>* bg_inds) const {
    int fg_offset = fg_inds->size();
    int bg_offset = bg_inds->size();
    for (int64_t i = 0; i < row; ++i) {
      T max_dist = -1;
      for (int64_t j = 0; j < col; ++j) {
        T val = dist_data[i * col + j];
        if (val > max_dist) max_dist = val;
      }
      for (int64_t j = 0; j < col; ++j) {
        T val = dist_data[i * col + j];
        if (val == max_dist) target_label_data[j] = 1;
      }
    }

    // Pick the fg/bg and count the number
    for (int64_t j = 0; j < col; ++j) {
      if (anchor_to_gt_max[j] > pos_threshold) {
        target_label_data[j] = 1;
      } else if (anchor_to_gt_max[j] < neg_threshold) {
        target_label_data[j] = 0;
      }
      if (target_label_data[j] == 1) {
        fg_inds->push_back(fg_offset + j);
      } else if (target_label_data[j] == 0) {
        bg_inds->push_back(bg_offset + j);
      }
    }
  }

  void RpnTargetAssign(const Tensor& dist, const float pos_threshold,
                       const float neg_threshold, const int rpn_batch_size,
                       const int fg_num, std::minstd_rand engine,
                       std::vector<int>* fg_inds, std::vector<int>* bg_inds,
                       int* target_label_data) const {
    auto* dist_data = dist.data<T>();
    int64_t row = dist.dims()[0];
    int64_t col = dist.dims()[1];
    int fg_offset = fg_inds->size();
    int bg_offset = bg_inds->size();
    std::vector<float> anchor_to_gt_max;
    // Calculate the max IoU between anchors and gt boxes
    AnchorToGt(dist_data, row, col, &anchor_to_gt_max);
    // Follow the Faster RCNN's implementation
    ScoreAssign(dist_data, anchor_to_gt_max, row, col, pos_threshold,
                neg_threshold, target_label_data, fg_inds, bg_inds);
    // Reservoir Sampling
    std::uniform_real_distribution<float> uniform(0, 1);
    if (fg_inds->size() > fg_num) {
      for (int i = fg_num; i < fg_inds->size(); ++i) {
        int rng_ind = std::floor(uniform(engine) * fg_num);
        if (rng_ind < fg_num)
          std::iter_swap(fg_inds->begin() + rng_ind + fg_offset,
                         fg_inds->begin() + i + fg_offset);
      }
    }
    int bg_num = rpn_batch_size - fg_inds->size();
    if (bg_inds->size() > bg_num) {
      for (int i = bg_num; i < bg_inds->size(); ++i) {
        int rng_ind = std::floor(uniform(engine) * bg_num);
        if (rng_ind < bg_num)
          std::iter_swap(bg_inds->begin() + rng_ind + bg_offset,
                         bg_inds->begin() + i + bg_offset);
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* match_dist = context.Input<LoDTensor>("Overlap");
    auto* loc_index = context.Output<Tensor>("LocationIndex");
    auto* score_index = context.Output<Tensor>("ScoreIndex");
    auto* tgt_lbl = context.Output<Tensor>("TargetLabel");

    auto col = match_dist->dims()[1];
    int64_t n = match_dist->lod().size() == 0UL
                    ? 1
                    : static_cast<int64_t>(match_dist->lod().back().size() - 1);
    if (match_dist->lod().size()) {
      PADDLE_ENFORCE_EQ(match_dist->lod().size(), 1UL,
                        "Only support 1 level of LoD.");
    }
    int rpn_batch_size = context.Attr<int>("rpn_batch_size_per_im");
    float pos_threshold = context.Attr<float>("rpn_positive_overlap");
    float neg_threshold = context.Attr<float>("rpn_negative_overlap");
    float fg_fraction = context.Attr<float>("fg_fraction");

    int fg_num = static_cast<int>(rpn_batch_size * fg_fraction);

    int* target_label_data =
        tgt_lbl->mutable_data<int>({n * col, 1}, context.GetPlace());

    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, int> iset;
    iset(dev_ctx, tgt_lbl, static_cast<int>(-1));

    std::vector<int> fg_inds;
    std::vector<int> bg_inds;
    std::random_device rnd;
    std::minstd_rand engine;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
    engine.seed(seed);

    if (n == 1) {
      RpnTargetAssign(*match_dist, pos_threshold, neg_threshold, rpn_batch_size,
                      fg_num, engine, &fg_inds, &bg_inds, target_label_data);
    } else {
      auto lod = match_dist->lod().back();
      for (size_t i = 0; i < lod.size() - 1; ++i) {
        Tensor one_ins = match_dist->Slice(lod[i], lod[i + 1]);
        RpnTargetAssign(one_ins, pos_threshold, neg_threshold, rpn_batch_size,
                        fg_num, engine, &fg_inds, &bg_inds,
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
    AddInput("Overlap", "");
    AddAttr<float>("rpn_positive_overlap", "").SetDefault(0.7);
    AddAttr<float>("rpn_negative_overlap", "").SetDefault(0.3);
    AddAttr<float>("fg_fraction", "").SetDefault(0.25);
    AddAttr<int>("rpn_batch_size_per_im", "").SetDefault(256);
    AddAttr<bool>("fix_seed", "").SetDefault(false);
    AddAttr<int>("seed", "").SetDefault(0);
    AddOutput("LocationIndex", "");
    AddOutput("ScoreIndex", "");
    AddOutput("TargetLabel", "");
    AddComment(R"DOC(
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
