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

template <typename DeviceContext, typename T>
class TargetConfidenceAssignOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_conf = ctx.Input<framework::Tensor>("Conf");
    auto* in_gt_labels = ctx.Input<framework::LoDTensor>("GTLabels");
    auto* in_match_indices = ctx.Input<framework::LoDTensor>("MatchIndices");
    auto* in_neg_indices = ctx.Input<framework::LoDTensor>("NegIndices");

    auto* out_conf_gt = ctx.Output<framework::LoDTensor>("ConfGT");
    auto* out_conf_pred = ctx.Output<framework::LoDTensor>("ConfPred");
    int background_label_id = ctx.Attr<int>("background_label_id");

    auto in_conf_dim = in_conf->dims();
    auto gt_lod = in_gt_labels->lod();
    auto neg_indices_lod = in_neg_indices->lod();
    int batch_size = in_conf_dim[0];
    int prior_num = in_conf_dim[1];
    int class_num = in_conf_dim[2];

    auto conf = framework::EigenTensor<T, 3>::From(*in_conf);
    auto gt_labels = framework::EigenTensor<int, 2>::From(*in_gt_labels);
    auto match_indices =
        framework::EigenTensor<int, 2>::From(*in_match_indices);
    auto neg_indices = framework::EigenTensor<int, 2>::From(*in_neg_indices);

    int match_num = 0;
    int neg_num = in_neg_indices->dims()[0];
    for (int n = 0; n < batch_size; ++n) {
      for (int p = 0; p < prior_num; ++p) {
        if (match_indices(n, p) != -1) match_num++;
      }
    }

    framework::LoD out_lod;
    out_lod.resize(1);
    out_lod[0].push_back(0);
    out_conf_gt->mutable_data<int>(
        framework::make_ddim({match_num + neg_num, 1}), ctx.GetPlace());
    out_conf_pred->mutable_data<T>(
        framework::make_ddim({match_num + neg_num, class_num}), ctx.GetPlace());

    auto conf_gt = framework::EigenTensor<int, 2>::From(*out_conf_gt);
    auto conf_pred = framework::EigenTensor<T, 2>::From(*out_conf_pred);

    int count = 0;
    for (int n = 0; n < batch_size; ++n) {
      for (int p = 0; p < prior_num; ++p) {
        int idx = match_indices(n, p);
        if (idx == -1) continue;
        int gt_start = gt_lod[0][n];
        int gt_offset = gt_start + idx;
        int label = gt_labels(gt_offset);
        conf_gt(count) = label;
        for (int c = 0; c < class_num; ++c) {
          conf_pred(count, c) = conf(n, p, c);
        }
        count += 1;
      }

      int neg_start = neg_indices_lod[0][n];
      int neg_end = neg_indices_lod[0][n + 1];
      for (int ne = neg_start; ne < neg_end; ++ne) {
        int idx = neg_indices(ne);
        conf_gt(count) = background_label_id;
        for (int c = 0; c < class_num; ++c) {
          conf_pred(count, c) = conf(n, idx, c);
        }
        count += 1;
      }
      out_lod[0].push_back(count);
    }
    out_conf_gt->set_lod(out_lod);
    out_conf_pred->set_lod(out_lod);
  }
};

}  // namespace operators
}  // namespace paddle
