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

class GenerateProposalLabelsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("DistMat"),
        "Input(DistMat) of GenerateProposalLabelsOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasInput("GtClasses"),
        "Input(GtClasses) of GenerateProposalLabelsOp should not be null");

    PADDLE_ENFORCE(
        ctx->HasOutput("FGIndex"),
        "Output(FGIndex) of GenerateProposalLabelsOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("BGIndex"),
        "Output(BGIndex) of GenerateProposalLabelsOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("LabelInt32"),
        "Output(LabelInt32) of GenerateProposalLabelsOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasOutput("GTIndex"),
        "Output(GTIndex) of GenerateProposalLabelsOp should not be null");

    auto in_dims = ctx->GetInputDim("DistMat");
    PADDLE_ENFORCE_EQ(in_dims.size(), 2,
                      "The rank of Input(DistMat) must be 2.");
  }
};

template <typename T>
class GenerateProposalLabelsKernel : public framework::OpKernel<T> {
 public:
  void ReservoirSampling(const int num, const int offset,
                         std::minstd_rand engine,
                         std::vector<int>* inds) const {
    std::uniform_real_distribution<float> uniform(0, 1);
    const int64_t size = static_cast<int64_t>(inds->size() - offset);
    if (size > num) {
      for (int64_t i = num; i < size; ++i) {
        int rng_ind = std::floor(uniform(engine) * i);
        if (rng_ind < num)
          std::iter_swap(inds->begin() + rng_ind + offset,
                         inds->begin() + i + offset);
      }
    }
  }

  void ReservoirSampling(const int num, const int offset,
                         std::minstd_rand engine, std::vector<int>* inds_a,
                         std::vector<int>* inds_b) const {
    std::uniform_real_distribution<float> uniform(0, 1);
    const int64_t size = static_cast<int64_t>(inds->size() - offset);
    if (size > num) {
      for (int64_t i = num; i < size; ++i) {
        int rng_ind = std::floor(uniform(engine) * i);
        if (rng_ind < num) {
          std::iter_swap(inds_a->begin() + rng_ind + offset,
                         inds_a->begin() + i + offset);
          std::iter_swap(inds_b->begin() + rng_ind + offset,
                         inds_b->begin() + i + offset);
        }
      }
    }
  }

  void SampleRoisForOneImage(const framework::ExecutionContext& ctx,
                             const Tensor& dist, const int batch_size_per_im,
                             const float fg_fraction, const float fg_thresh,
                             const float bg_thresh_hi, const float bg_thresh_lo,
                             std::minstd_rand engine, std::vector<int>* fg_inds,
                             std::vector<int>* bg_inds,
                             std::vector<int>* gt_inds) const {
    auto* dist_data = dist.data<T>();
    int64_t row = dist.dims()[0];
    int64_t col = dist.dims()[1];
    int fg_offset = fg_inds->size();
    int bg_offset = bg_inds->size();

    // Calculate the max IoU between anchors and gt boxes
    Tensor overlaps_max;
    overlaps_max.mutable_data<T>(
        framework::make_ddim({static_cast<int64_t>(col), 1}),
        platform::CPUPlace());
    auto& place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    auto proposal_to_gt_overlaps = EigenMatrix<T>::From(dist);
    auto col_max = EigenMatrix<T>::From(overlaps_max);
    col_max.device(place) =
        proposal_to_gt_overlaps.maximum(Eigen::DSizes<int, 1>(0))
            .reshape(Eigen::DSizes<int, 2>(static_cast<int64_t>(col), 1));
    // Follow the Faster RCNN's implementation
    for (int64_t i = 0; i < row; ++i) {
      const T* v = dist_data + i * col;
      T max_dist = *std::max_element(v, v + col);
      if (max_dist > fg_thresh) {
        for (int64_t j = 0; j < col; ++j) {
          T val = dist_data[i * col + j];
          if (val == max_dist) {
            fg_inds.emplace_back(i);
            gt_inds.emplace_back(j);
          }
        }
      } else {
        if ((max_dist >= bg_thresh_lo) && (max_dist < bg_thresh_hi)) {
          bg_inds.emplace_back(i);
        }
      }
    }

    // Reservoir Sampling
    fg_rois_per_im = std::floor(batch_size_per_im * fg_fraction);
    fg_rois_this_image = fg_inds->size() - fg_offset;
    fg_rois_per_this_image = std::min(fg_rois_per_im, fg_rois_this_image);
    ReservoirSampling(fg_rois_per_this_image, fg_offset, engine, fg_inds,
                      gt_inds);

    bg_rois_per_image = batch_size_per_im - fg_rois_per_this_image;
    bg_rois_this_image = bg_inds->size() - bg_offset;
    bg_rois_per_this_image = std::min(bg_rois_per_image, bg_rois_this_image);
    ReservoirSampling(bg_rois_per_this_image, bg_offset, engine, bg_inds);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* dist = context.Input<LoDTensor>("DistMat");
    auto* fg_index = context.Output<Tensor>("FGIndex");
    auto* bg_index = context.Output<Tensor>("BGIndex");
    auto* gt_index = context.Output<Tensor>("GTIndex");

    auto col = dist->dims()[1];
    int64_t n = dist->lod().size() == 0UL
                    ? 1
                    : static_cast<int64_t>(dist->lod().back().size() - 1);
    if (dist->lod().size()) {
      PADDLE_ENFORCE_EQ(dist->lod().size(), 1UL,
                        "Only support 1 level of LoD.");
    }

    int batch_size_per_im = context.Attr<int>("batch_size_per_im");
    float fg_fraction = context.Attr<int>("fg_fraction");
    float fg_thresh = context.Attr<float>("fg_thresh");
    float bg_thresh_hi = context.Attr<float>("bg_thresh_hi");
    float bg_thresh_lo = context.Attr<float>("bg_thresh_lo");

    std::vector<int> fg_inds;
    std::vector<int> bg_inds;
    std::vector<int> gt_inds;
    std::random_device rnd;
    std::minstd_rand engine;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
    engine.seed(seed);

    if (n == 1) {
      SampleRoisForOneImage(context, *dist, batch_size_per_im, fg_fraction,
                            fg_thresh, bg_thresh_hi, bg_thresh_lo, engine,
                            &fg_inds, &bg_inds, &gt_inds);
    } else {
      auto lod = dist->lod().back();
      for (size_t i = 0; i < lod.size() - 1; ++i) {
        Tensor one_ins = dist->Slice(lod[i], lod[i + 1]);
        SampleRoisForOneImage(context, one_ins, batch_size_per_im, fg_fraction,
                              fg_thresh, bg_thresh_hi, bg_thresh_lo, engine,
                              &fg_inds, &bg_inds, &gt_inds);
      }
    }
    int* fg_index_data = fg_index->mutable_data<int>(
        {static_cast<int>(fg_inds.size())}, context.GetPlace());
    int* bg_index_data = bg_index->mutable_data<int>(
        {static_cast<int>(bg_inds.size())}, context.GetPlace());
    int* gt_index_data = gt_index->mutable_data<int>(
        {static_cast<int>(gt_inds.size())}, context.GetPlace());
    memcpy(fg_index_data, reinterpret_cast<int*>(&fg_inds[0]),
           fg_inds.size() * sizeof(int));
    memcpy(bg_index_data, reinterpret_cast<int*>(&bg_inds[0]),
           bg_inds.size() * sizeof(int));
    memcpy(gt_index_data, reinterpret_cast<int*>(&gt_inds[0]),
           gt_inds.size() * sizeof(int));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(generate_proposal_labels, ops::GenerateProposalLabelsOp,
                  ops::GenerateProposalLabelsOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(generate_proposal_labels,
                       ops::GenerateProposalLabelsKernel<float>,
                       ops::GenerateProposalLabelsKernel<double>);
