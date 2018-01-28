/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
inline void EncodeBox(std::vector<T>& prior_box, std::vector<T>& prior_variance,
                      std::vector<T>& gt_box, bool encode_variance_in_target,
                      std::vector<T>& encode_box) {
  T prior_width = prior_box[2] - prior_box[0];
  T prior_height = prior_box[3] - prior_box[1];
  T prior_center_x = (prior_box[0] + prior_box[2]) / 2.0;
  T prior_center_y = (prior_box[1] + prior_box[3]) / 2.0;

  T gt_width = gt_box[2] - gt_box[0];
  T gt_height = gt_box[3] - gt_box[1];
  T gt_center_x = (gt_box[0] + gt_box[2]) / 2.0;
  T gt_center_y = (gt_box[1] + gt_box[3]) / 2.0;

  encode_box.resize(4);
  encode_box[0] = (gt_center_x - prior_center_x) / prior_width;
  encode_box[1] = (gt_center_y - prior_center_y) / prior_height;
  encode_box[2] = log(gt_width / prior_width);
  encode_box[3] = log(gt_height / prior_height);
  return;
}

template <typename DeviceContext, typename T>
class TargetLocationAssignOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_loc = ctx.Input<framework::Tensor>("Loc");
    auto* in_encode_gt_boxes = ctx.Input<framework::LoDTensor>("GTBoxes");
    auto* in_match_indices = ctx.Input<framework::LoDTensor>("MatchIndices");
    auto* in_prior_boxes = ctx.Input<framework::Tensor>("PriorBoxes");
    auto* in_prior_variances = ctx.Input<framework::Tensor>("PriorVariances");

    bool encode_variance_in_target =
        ctx.Attr<bool>("encode_variance_in_target");

    auto* out_loc_gt = ctx.Output<framework::LoDTensor>("LocGT");
    auto* out_loc_pred = ctx.Output<framework::LoDTensor>("LocPred");

    auto in_loc_dim = in_loc->dims();
    auto gt_lod = in_encode_gt_boxes->lod();
    int batch_size = in_loc_dim[0];
    int prior_num = in_loc_dim[1];

    auto loc = framework::EigenTensor<T, 3>::From(*in_loc);
    auto encode_gt_boxes =
        framework::EigenTensor<T, 3>::From(*in_encode_gt_boxes);
    auto match_indices =
        framework::EigenTensor<int, 2>::From(*in_match_indices);
    auto prior_boxes = framework::EigenTensor<T, 2>::From(*in_prior_boxes);
    auto prior_variances =
        framework::EigenTensor<T, 2>::From(*in_prior_variances);

    int output_size = 0;
    for (int n = 0; n < batch_size; ++n) {
      for (int p = 0; p < prior_num; ++p) {
        if (match_indices(n, p) != -1) output_size++;
      }
    }
    out_loc_gt->mutable_data<T>(framework::make_ddim({output_size, 4}),
                                ctx.GetPlace());
    out_loc_pred->mutable_data<T>(framework::make_ddim({output_size, 4}),
                                  ctx.GetPlace());
    auto loc_gt = framework::EigenTensor<T, 2>::From(*out_loc_gt);
    auto loc_pred = framework::EigenTensor<T, 2>::From(*out_loc_pred);

    std::vector<T> prior_box(4);
    std::vector<T> prior_variance(4);
    std::vector<T> gt_box(4);
    framework::LoD out_lod;
    out_lod.resize(1);
    int count = 0;
    out_lod[0].push_back(count);
    for (int n = 0; n < batch_size; ++n) {
      for (int p = 0; p < prior_num; ++p) {
        int idx = match_indices(n, p);
        if (idx == -1) continue;
        int gt_start = gt_lod[0][n];
        int gt_end = gt_lod[0][n + 1];
        int gt_offset = gt_start + idx;
        PADDLE_ENFORCE_LE(gt_offset, gt_end,
                          "The matched box index is larger than max size");
        for (int k = 0; k < 4; ++k) {
          prior_box[k] = prior_boxes(p, k);
          prior_variance[k] = prior_variances(p, k);
          gt_box[k] = encode_gt_boxes(gt_offset, p, k);
          loc_pred(count, k) = loc(n, p, k);
        }
        std::vector<T> gt_encode(4);
        std::vector<T> loc_pred_out(4);
        EncodeBox<T>(prior_box, prior_variance, gt_box,
                     encode_variance_in_target, gt_encode);
        for (int k = 0; k < 4; ++k) {
          loc_gt(count, k) = gt_encode[k] / prior_variance[k];
        }
        if (encode_variance_in_target) {
          for (int k = 0; k < 4; ++k) {
            loc_pred(count, k) = loc_pred(count, k) / prior_variance[k];
          }
        }
        count++;
      }
      out_lod[0].push_back(count);
    }
    out_loc_gt->set_lod(out_lod);
    out_loc_pred->set_lod(out_lod);
  }
};  // namespace operators

}  // namespace operators
}  // namespace paddle
