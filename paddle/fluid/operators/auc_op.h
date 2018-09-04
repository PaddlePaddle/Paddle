/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class AucKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *predict = ctx.Input<Tensor>("Predict");
    auto *label = ctx.Input<Tensor>("Label");

    std::string curve = ctx.Attr<std::string>("curve");
    int num_thresholds = ctx.Attr<int>("num_thresholds");
    int num_pred_buckets = num_thresholds + 1;

    // Only use output var for now, make sure it's persistable and
    // not cleaned up for each batch.
    auto *auc = ctx.Output<Tensor>("AUC");
    auto *stat_pos = ctx.Output<Tensor>("StatPosOut");
    auto *stat_neg = ctx.Output<Tensor>("StatNegOut");

    auto *stat_pos_data = stat_pos->mutable_data<int64_t>(ctx.GetPlace());
    auto *stat_neg_data = stat_neg->mutable_data<int64_t>(ctx.GetPlace());
    calcAuc(ctx, label, predict, stat_pos_data, stat_neg_data, num_thresholds,
            auc);

    auto *batch_auc = ctx.Output<Tensor>("BatchAUC");
    std::vector<int64_t> stat_pos_batch(num_pred_buckets, 0);
    std::vector<int64_t> stat_neg_batch(num_pred_buckets, 0);
    calcAuc(ctx, label, predict, stat_pos_batch.data(), stat_neg_batch.data(),
            num_thresholds, batch_auc);
  }

 private:
  inline static double trapezoidArea(double X1, double X2, double Y1,
                                     double Y2) {
    return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
  }

  inline static void calcAuc(const framework::ExecutionContext &ctx,
                             const framework::Tensor *label,
                             const framework::Tensor *predict,
                             int64_t *stat_pos, int64_t *stat_neg,
                             int num_thresholds,
                             framework::Tensor *auc_tensor) {
    size_t batch_size = predict->dims()[0];
    size_t inference_width = predict->dims()[1];
    const T *inference_data = predict->data<T>();
    const auto *label_data = label->data<int64_t>();

    auto *auc = auc_tensor->mutable_data<double>(ctx.GetPlace());

    for (size_t i = 0; i < batch_size; i++) {
      uint32_t binIdx = static_cast<uint32_t>(
          inference_data[i * inference_width + 1] * num_thresholds);
      if (label_data[i]) {
        stat_pos[binIdx] += 1.0;
      } else {
        stat_neg[binIdx] += 1.0;
      }
    }

    *auc = 0.0f;

    double totPos = 0.0;
    double totNeg = 0.0;
    double totPosPrev = 0.0;
    double totNegPrev = 0.0;

    int idx = num_thresholds;

    while (idx >= 0) {
      totPosPrev = totPos;
      totNegPrev = totNeg;
      totPos += stat_pos[idx];
      totNeg += stat_neg[idx];
      *auc += trapezoidArea(totNeg, totNegPrev, totPos, totPosPrev);

      --idx;
    }

    if (totPos > 0.0 && totNeg > 0.0) {
      *auc = *auc / totPos / totNeg;
    }
  }
};

}  // namespace operators
}  // namespace paddle
