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
#include "paddle/fluid/framework/eigen.h"
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
    auto *auc = ctx.Output<Tensor>("AUC");
    // Only use output var for now, make sure it's persistable and
    // not cleaned up for each batch.
    auto *stat_pos = ctx.Output<Tensor>("StatPosOut");
    auto *stat_neg = ctx.Output<Tensor>("StatNegOut");
    auto *auc_data = auc->mutable_data<double>(ctx.GetPlace());

    std::string curve = ctx.Attr<std::string>("curve");
    int num_thresholds = ctx.Attr<int>("num_thresholds");

    size_t batch_size = predict->dims()[0];
    size_t inference_width = predict->dims()[1];

    const T *inference_data = predict->data<T>();
    const auto *label_data = label->data<int64_t>();

    auto *stat_pos_data = stat_pos->mutable_data<int64_t>(ctx.GetPlace());
    auto *stat_neg_data = stat_neg->mutable_data<int64_t>(ctx.GetPlace());

    for (size_t i = 0; i < batch_size; i++) {
      uint32_t binIdx = static_cast<uint32_t>(
          inference_data[i * inference_width + 1] * num_thresholds);
      if (label_data[i]) {
        stat_pos_data[binIdx] += 1.0;
      } else {
        stat_neg_data[binIdx] += 1.0;
      }
    }

    *auc_data = 0.0f;

    double totPos = 0.0;
    double totNeg = 0.0;
    double totPosPrev = 0.0;
    double totNegPrev = 0.0;

    int idx = num_thresholds;

    while (idx >= 0) {
      totPosPrev = totPos;
      totNegPrev = totNeg;
      totPos += stat_pos_data[idx];
      totNeg += stat_neg_data[idx];
      *auc_data += trapezoidArea(totNeg, totNegPrev, totPos, totPosPrev);
      --idx;
    }

    if (totPos > 0.0 && totNeg > 0.0) {
      *auc_data = *auc_data / totPos / totNeg;
    }
  }

 private:
  inline static double trapezoidArea(double X1, double X2, double Y1,
                                     double Y2) {
    return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
  }
};

}  // namespace operators
}  // namespace paddle
