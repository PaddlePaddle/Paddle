/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/framework/op_registry.h"
#include "paddle/platform/assert.h"
#include "paddle/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct TargetAssignFunctor {
  const T* gt_box_;
  const int* gt_label_;
  const int* match_indices_;
  const size_t* lod_;
  const int background_label_;
  const int64_t num_;
  const int64_t num_prior_box_;

  T* out_box_;
  T* out_box_wt_;
  int* out_label_;
  T* out_label_wt_;

  TargetAssignFunctor(const T* gt_box, const int* gt_label,
                      const int* match_indices, const size_t* lod,
                      const int background_label, const int64_t num,
                      const int64_t np, T* out_box, T* out_box_wt,
                      int* out_label, T* out_label_wt)
      : gt_box_(gt_box),
        gt_label_(gt_label),
        match_indices_(match_indices),
        lod_(lod),
        background_label_(background_label),
        num_(num),
        num_prior_box_(np),
        out_box_(out_box),
        out_box_wt_(out_box_wt),
        out_label_(out_label),
        out_label_wt_(out_label_wt) {}

  HOSTDEVICE void operator()(size_t i) const {
    int row = i / num_prior_box_;
    int col = i - row * num_prior_box_;

    size_t row_off = lod_[row];
    int offset = row * num_prior_box_ + col;

    int id = match_indices_[offset];
    T* obox = out_box_ + offset * 4;
    int* olabel = out_label_ + offset;
    T* obox_wt = out_box_wt_ + offset;
    T* olabel_wt = out_label_wt_ + offset;

    if (id > -1) {
      const T* gtbox = gt_box_ + ((row_off + id) * num_prior_box_ + col) * 4;

      obox[0] = gtbox[0];
      obox[1] = gtbox[1];
      obox[2] = gtbox[2];
      obox[3] = gtbox[3];

      olabel[0] = gt_label_[row_off + id];
      obox_wt[0] = static_cast<T>(1.);
      olabel_wt[0] = static_cast<T>(1.);
    } else {
      obox[0] = static_cast<T>(0.);
      obox[1] = static_cast<T>(0.);
      obox[2] = static_cast<T>(0.);
      obox[3] = static_cast<T>(0.);

      olabel[0] = background_label_;
      obox_wt[0] = static_cast<T>(0.);
      olabel_wt[0] = static_cast<T>(0.);
    }
  }
};

template <typename DeviceContext, typename T>
struct NegTargetAssignFunctor {
  void operator()(const platform::DeviceContext& ctx, const int* neg_indices,
                  const size_t* lod, const int num, const int num_prior_box,
                  const int background_label, int* out_label,
                  T* out_label_wt) const;
};

template <typename DeviceContext, typename T>
class TargetAssignKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* enc_gt_box = ctx.Input<framework::LoDTensor>("EncodedGTBBox");
    auto* gt_label = ctx.Input<framework::LoDTensor>("GTScoreLabel");
    auto* match_indices = ctx.Input<framework::Tensor>("MatchIndices");
    auto* neg_indices = ctx.Input<framework::LoDTensor>("NegIndices");

    auto* out_box = ctx.Output<framework::Tensor>("PredBBoxLabel");
    auto* out_box_wt = ctx.Output<framework::Tensor>("PredBBoxWeight");
    auto* out_label = ctx.Output<framework::Tensor>("PredScoreLabel");
    auto* out_label_wt = ctx.Output<framework::Tensor>("PredScoreWeight");

    PADDLE_ENFORCE_EQ(enc_gt_box->lod().size(), 1UL);
    PADDLE_ENFORCE_EQ(gt_label->lod().size(), 1UL);
    PADDLE_ENFORCE_EQ(neg_indices->lod().size(), 1UL);

    int background_label = ctx.Attr<int>("background_label");

    const T* box_data = enc_gt_box->data<T>();
    const int* label_data = gt_label->data<int>();
    const int* match_idx_data = match_indices->data<int>();
    const int* neg_idx_data = neg_indices->data<int>();

    T* obox_data = out_box->mutable_data<T>(ctx.GetPlace());
    T* obox_wt_data = out_box_wt->mutable_data<T>(ctx.GetPlace());
    int* olabel_data = out_label->mutable_data<int>(ctx.GetPlace());
    T* olabel_wt_data = out_label_wt->mutable_data<T>(ctx.GetPlace());

    int64_t num = match_indices->dims()[0];
    int64_t num_prior_box = match_indices->dims()[1];

    auto gt_lod = enc_gt_box->lod().back();
    auto gt_label_lod = gt_label->lod().back();
    auto neg_lod = neg_indices->lod().back();
    for (size_t i = 0; i < gt_lod.size(); ++i) {
      PADDLE_ENFORCE_EQ(gt_lod.data()[i], gt_label_lod.data()[i]);
    }

    size_t* gt_lod_data = gt_lod.MutableData(ctx.GetPlace());
    size_t* neg_lod_data = neg_lod.MutableData(ctx.GetPlace());

    TargetAssignFunctor<T> functor(box_data, label_data, match_idx_data,
                                   gt_lod_data, background_label, num,
                                   num_prior_box, obox_data, obox_wt_data,
                                   olabel_data, olabel_wt_data);

    auto& device_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(device_ctx,
                                                num * num_prior_box);
    for_range(functor);

    NegTargetAssignFunctor<DeviceContext, T> neg_trg_functor;
    neg_trg_functor(device_ctx, neg_idx_data, neg_lod_data, num, num_prior_box,
                    background_label, olabel_data, olabel_wt_data);
  }
};

}  // namespace operators
}  // namespace paddle
