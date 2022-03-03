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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

enum MiningType { kNone = 0, kMaxNegative, kHardExample };

template <typename T>
bool SortScoreDescend(const std::pair<float, T>& pair1,
                      const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

inline bool IsEligibleMining(const MiningType mining_type, const int match_idx,
                             const float match_dist,
                             const float neg_dist_threshold) {
  if (mining_type == MiningType::kMaxNegative) {
    return match_idx == -1 && match_dist < neg_dist_threshold;
  } else if (mining_type == MiningType::kHardExample) {
    return true;
  } else {
    return false;
  }
}

inline MiningType GetMiningType(std::string str) {
  if (str == "max_negative") {
    return MiningType::kMaxNegative;
  } else if (str == "hard_example") {
    return MiningType::kHardExample;
  } else {
    return MiningType::kNone;
  }
}

template <typename DeviceContext, typename T>
class MineHardExamplesKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_cls_loss = ctx.Input<framework::Tensor>("ClsLoss");
    auto* in_loc_loss = ctx.Input<framework::Tensor>("LocLoss");
    auto* in_matched_indices = ctx.Input<framework::Tensor>("MatchIndices");
    auto* in_match_dist = ctx.Input<framework::Tensor>("MatchDist");
    float neg_pos_ratio = ctx.Attr<float>("neg_pos_ratio");
    T neg_dist_threshold =
        static_cast<T>(ctx.Attr<float>("neg_dist_threshold"));
    int sample_size = ctx.Attr<int>("sample_size");
    MiningType mining_type =
        GetMiningType(ctx.Attr<std::string>("mining_type"));

    auto out_neg_indices = ctx.Output<framework::LoDTensor>("NegIndices");
    auto out_match_indices =
        ctx.Output<framework::Tensor>("UpdatedMatchIndices");

    framework::TensorCopy(*in_matched_indices, ctx.GetPlace(),
                          out_match_indices);

    int batch_size = in_matched_indices->dims()[0];
    int prior_num = in_matched_indices->dims()[1];

    auto match_indices = framework::EigenMatrix<int>::From(*in_matched_indices);

    auto match_indices_et =
        framework::EigenMatrix<int>::From(*out_match_indices);

    auto match_dist = framework::EigenMatrix<T>::From(*in_match_dist);

    const T* cls_loss = in_cls_loss->data<T>();
    const T* loc_loss = nullptr;
    if (in_loc_loss) {
      loc_loss = in_loc_loss->data<T>();
    }

    std::vector<std::vector<int>> all_neg_indices;
    std::vector<size_t> batch_starts = {0};
    for (int n = 0; n < batch_size; ++n) {
      std::vector<std::pair<T, size_t>> loss_idx;
      int neg_sel = 0;
      for (int m = 0; m < prior_num; ++m) {
        if (IsEligibleMining(mining_type, match_indices(n, m), match_dist(n, m),
                             neg_dist_threshold)) {
          T loss = cls_loss[n * prior_num + m];
          if (mining_type == MiningType::kHardExample && loc_loss != nullptr) {
            loss = cls_loss[n * prior_num + m] + loc_loss[n * prior_num + m];
          }
          loss_idx.push_back(std::make_pair(loss, m));
          ++neg_sel;
        }
      }

      if (mining_type == MiningType::kMaxNegative) {
        int num_pos = 0;
        for (int m = 0; m < prior_num; ++m) {
          if (match_indices(n, m) != -1) ++num_pos;
        }
        neg_sel = std::min(static_cast<int>(num_pos * neg_pos_ratio), neg_sel);
      } else if (mining_type == MiningType::kHardExample) {
        neg_sel = std::min(sample_size, neg_sel);
      }

      std::sort(loss_idx.begin(), loss_idx.end(), SortScoreDescend<size_t>);
      std::set<int> sel_indices;
      std::vector<int> neg_indices;
      std::transform(loss_idx.begin(), loss_idx.begin() + neg_sel,
                     std::inserter(sel_indices, sel_indices.begin()),
                     [](std::pair<T, size_t>& l) -> int {
                       return static_cast<int>(l.second);
                     });

      if (mining_type == MiningType::kHardExample) {
        for (int m = 0; m < prior_num; ++m) {
          if (match_indices(n, m) > -1) {
            if (sel_indices.find(m) == sel_indices.end()) {
              match_indices_et(n, m) = -1;
            }
          } else {
            if (sel_indices.find(m) != sel_indices.end()) {
              neg_indices.push_back(m);
            }
          }
        }
      } else {
        neg_indices.resize(sel_indices.size());
        std::copy(sel_indices.begin(), sel_indices.end(), neg_indices.begin());
      }

      all_neg_indices.push_back(neg_indices);
      batch_starts.push_back(batch_starts.back() + neg_indices.size());
    }

    framework::LoD out_neg_indices_lod;
    out_neg_indices_lod.emplace_back(batch_starts);
    int neg_offset = 0;
    auto neg_data = out_neg_indices->mutable_data<int>(
        phi::make_ddim({static_cast<int>(batch_starts.back()), 1}),
        ctx.GetPlace());

    for (auto neg_indices : all_neg_indices) {
      std::copy(neg_indices.begin(), neg_indices.end(), neg_data + neg_offset);
      neg_offset += neg_indices.size();
    }
    out_neg_indices->set_lod(out_neg_indices_lod);
    return;
  }
};

class MineHardExamplesOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("ClsLoss"), "Input", "ClsLoss",
                   "mine_hard_examples");
    OP_INOUT_CHECK(ctx->HasInput("MatchIndices"), "Input", "MatchIndices",
                   "mine_hard_examples");
    OP_INOUT_CHECK(ctx->HasInput("MatchDist"), "Input", "MatchDist",
                   "mine_hard_examples");
    OP_INOUT_CHECK(ctx->HasOutput("NegIndices"), "Output", "NegIndices",
                   "mine_hard_examples");
    OP_INOUT_CHECK(ctx->HasOutput("UpdatedMatchIndices"), "Output",
                   "UpdatedMatchIndices", "mine_hard_examples");

    auto cls_loss_dims = ctx->GetInputDim("ClsLoss");
    auto idx_dims = ctx->GetInputDim("MatchIndices");
    auto dis_dims = ctx->GetInputDim("MatchDist");

    PADDLE_ENFORCE_EQ(cls_loss_dims.size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The shape of ClsLoss is [N, Np]. But received %d.",
                          cls_loss_dims.size()));
    PADDLE_ENFORCE_EQ(
        idx_dims.size(), 2UL,
        platform::errors::InvalidArgument(
            "The shape of MatchIndices is [N, Np]. But received %d.",
            idx_dims.size()));
    PADDLE_ENFORCE_EQ(dis_dims.size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The shape of MatchDist is [N, Np]. But received %d.",
                          dis_dims.size()));

    if (ctx->HasInput("LocLoss")) {
      auto loc_loss_dims = ctx->GetInputDim("LocLoss");
      PADDLE_ENFORCE_EQ(loc_loss_dims.size(), 2UL,
                        platform::errors::InvalidArgument(
                            "The shape of LocLoss is [N, Np]. But received %d.",
                            loc_loss_dims.size()));
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(cls_loss_dims[0], loc_loss_dims[0],
                          platform::errors::InvalidArgument(
                              "Batch size of ClsLoss and LocLoss must be the "
                              "same. But received batch size of ClsLoss was "
                              "%d, batch size of LocLoss was %d.",
                              cls_loss_dims[0], loc_loss_dims[0]));
        PADDLE_ENFORCE_EQ(cls_loss_dims[1], loc_loss_dims[1],
                          platform::errors::InvalidArgument(
                              "Prior box number of ClsLoss and LocLoss must be "
                              "the same. But received box number of ClsLoss "
                              "was %d, box number of LocLoss was %d.",
                              cls_loss_dims[1], loc_loss_dims[1]));
      }
    }

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(cls_loss_dims[0], idx_dims[0],
                        platform::errors::InvalidArgument(
                            "Batch size of ClsLoss and MatchIndices must be "
                            "the same. But received batch size of ClsLoss was "
                            "%d, batch size of MatchIndices was %d.",
                            cls_loss_dims[0], idx_dims[0]));
      PADDLE_ENFORCE_EQ(
          cls_loss_dims[1], idx_dims[1],
          platform::errors::InvalidArgument(
              "Prior box number of ClsLoss and "
              "MatchIndices must be the same. But received box number of "
              "ClsLoss was %d, box number of MatchIndices was %d.",
              cls_loss_dims[1], idx_dims[1]));

      PADDLE_ENFORCE_EQ(cls_loss_dims[0], dis_dims[0],
                        platform::errors::InvalidArgument(
                            "Batch size of ClsLoss and MatchDist must be the "
                            "same. But received batch size of ClsLoss was %d, "
                            "batch size of MatchDist was %d.",
                            cls_loss_dims[0], dis_dims[0]));
      PADDLE_ENFORCE_EQ(cls_loss_dims[1], idx_dims[1],
                        platform::errors::InvalidArgument(
                            "Prior box number of ClsLoss and MatchDist must be "
                            "the same. But received box number of ClsLoss was "
                            "%d, box number of MatchDist was %d.",
                            cls_loss_dims[1], idx_dims[1]));
    }

    auto mining_type =
        GetMiningType(ctx->Attrs().Get<std::string>("mining_type"));

    PADDLE_ENFORCE_NE(mining_type, MiningType::kNone,
                      platform::errors::InvalidArgument(
                          "mining_type must be hard_example or max_negative"));

    if (mining_type == MiningType::kMaxNegative) {
      auto neg_pos_ratio = ctx->Attrs().Get<float>("neg_pos_ratio");
      auto neg_dist_threshold = ctx->Attrs().Get<float>("neg_dist_threshold");
      PADDLE_ENFORCE_GT(neg_pos_ratio, 0.0f,
                        platform::errors::InvalidArgument(
                            "neg_pos_ratio must greater than zero in "
                            "max_negative mode. But received %f.",
                            neg_pos_ratio));
      PADDLE_ENFORCE_LT(neg_dist_threshold, 1.0f,
                        platform::errors::InvalidArgument(
                            "neg_dist_threshold must less than one in "
                            "max_negative mode. But received %f.",
                            neg_dist_threshold));
      PADDLE_ENFORCE_GT(neg_dist_threshold, 0.0f,
                        platform::errors::InvalidArgument(
                            "neg_dist_threshold must greater "
                            "than zero in max_negative mode. But received %f.",
                            neg_dist_threshold));
    } else if (mining_type == MiningType::kHardExample) {
      auto sample_size = ctx->Attrs().Get<int>("sample_size");
      PADDLE_ENFORCE_GT(sample_size, 0,
                        platform::errors::InvalidArgument(
                            "sample_size must greater than zero in "
                            "hard_example mode. But received %d.",
                            sample_size));
    }

    ctx->SetOutputDim("UpdatedMatchIndices", idx_dims);
    // The first dimension of NegIndices will be set correcttly in Compute.
    ctx->SetOutputDim("NegIndices", {-1, 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "ClsLoss"),
        platform::CPUPlace());
  }
};

class MineHardExamplesOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "ClsLoss",
        "(Tensor, default Tensor<float>), The classification loss with shape "
        "[N, Np], N is the batch size and Np is the number of prior box.");
    AddInput("LocLoss",
             "(Tensor, optional, default Tensor<float>), The localization loss "
             "with shape [N, Np], N is the batch size and Np is the number of "
             "prior box.")
        .AsDispensable();
    AddInput("MatchIndices",
             "(Tensor, Tensor<int>), Matched indices with shape [N, Np], N is "
             "the batch size and Np is the number of prior box. "
             "MatchIndices[i][j] equal -1 means the j-th prior box in i-th "
             "instance does not match any entity, otherwise means it is "
             "matched to row.");
    AddInput("MatchDist",
             "(Tensor, default Tensor<float>) Matched indices with shape [N, "
             "Np], N is the batch size and Np is the number of prior box.");
    AddAttr<float>("neg_pos_ratio",
                   "(float) The ratio of the negative box to the positive "
                   "box. Use only when mining_type is max_negative.")
        .SetDefault(1.0);
    AddAttr<float>("neg_dist_threshold",
                   "(float) The negative overlap upper bound for the unmatched "
                   "predictions. Use only when mining_type is max_negative.")
        .SetDefault(0.5);
    AddAttr<int>("sample_size",
                 "(float) The max sample size of negative box. Use only when "
                 "mining_type is hard_example.")
        .SetDefault(0);
    AddAttr<std::string>("mining_type",
                         "(float) The mining algorithm name, the value is "
                         "hard_example or max_negative.")
        .SetDefault("max_negative")
        .InEnum({"hard_example", "max_negative"});

    AddOutput(
        "NegIndices",
        "(LoDTensor<int>) The output of negative example indices. a LoDTensor "
        "with shape [Neg, 1]. The size of lod[0] minus 1 is batch size, "
        "and each element is the prior box index. "
        "For example, the batch size is 2, the lod is [[0, 1, 2]], "
        "the sample 0's box 1(MatchIndices[0][1]) is selected, "
        "and sample 1's box 0 is selected. The output NegIndices is "
        "[[1], [0]].");

    AddOutput("UpdatedMatchIndices",
              "(Tensor<int>) The output of updated MatchIndices, a tensor with "
              "shape [N, Np]. Only update when mining_type is "
              "hard_example. The input MatchIndices elements will be update to "
              "-1 when it is not in the candidate high loss list of negative "
              "examples.");

    AddComment(R"DOC(
Mine hard examples Operator.
This operator implements hard example mining to select a subset of negative box indices.
For each image, selects the box with highest losses. subject to the condition that the 
box cannot have an Matcht > neg_dist_threshold when mining_type is max_negative. 
The selected number is min(sample_size, max_negative_box_number) when mining_type is 
hard_example, or min(neg_pos_ratio * positive_box_number, max_negative_box_number) 
when mining_type is max_negative, where the max_negative_box_number is the count of 
MatchIndices elements with value -1.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    mine_hard_examples, ops::MineHardExamplesOp, ops::MineHardExamplesOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    mine_hard_examples,
    ops::MineHardExamplesKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MineHardExamplesKernel<paddle::platform::CPUDeviceContext, double>);
