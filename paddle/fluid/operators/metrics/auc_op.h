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
    // buckets contain numbers from 0 to num_thresholds
    int num_pred_buckets = num_thresholds + 1;
    int slide_steps = ctx.Attr<int>("slide_steps");

    // Only use output var for now, make sure it's persistable and
    // not cleaned up for each batch.
    auto *auc = ctx.Output<Tensor>("AUC");
    auto *stat_pos = ctx.Output<Tensor>("StatPosOut");
    auto *stat_neg = ctx.Output<Tensor>("StatNegOut");

    auto *origin_stat_pos = stat_pos->mutable_data<int64_t>(ctx.GetPlace());
    auto *origin_stat_neg = stat_neg->mutable_data<int64_t>(ctx.GetPlace());

    std::vector<int64_t> stat_pos_data(num_pred_buckets, 0);
    std::vector<int64_t> stat_neg_data(num_pred_buckets, 0);

    auto stat_pos_calc = stat_pos_data.data();
    auto stat_neg_calc = stat_neg_data.data();

    statAuc(label, predict, num_pred_buckets, num_thresholds, slide_steps,
            origin_stat_pos, origin_stat_neg, &stat_pos_calc, &stat_neg_calc);

    calcAuc(ctx, stat_pos_calc, stat_neg_calc, num_thresholds, auc);
  }

 private:
  inline static double trapezoidArea(double X1, double X2, double Y1,
                                     double Y2) {
    return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
  }

  inline static void statAuc(const framework::Tensor *label,
                             const framework::Tensor *predict,
                             const int num_pred_buckets,
                             const int num_thresholds, const int slide_steps,
                             int64_t *origin_stat_pos, int64_t *origin_stat_neg,
                             int64_t **stat_pos, int64_t **stat_neg) {
    size_t batch_size = predict->dims()[0];
    size_t inference_width = predict->dims()[1];
    const T *inference_data = predict->data<T>();
    const auto *label_data = label->data<int64_t>();

    for (size_t i = 0; i < batch_size; i++) {
      auto predict_data = inference_data[i * inference_width + 1];
      PADDLE_ENFORCE_LE(predict_data, 1,
                        "The predict data must less or equal 1.");
      PADDLE_ENFORCE_GE(predict_data, 0,
                        "The predict data must gather or equal 0.");

      uint32_t binIdx = static_cast<uint32_t>(predict_data * num_thresholds);
      if (label_data[i]) {
        (*stat_pos)[binIdx] += 1.0;
      } else {
        (*stat_neg)[binIdx] += 1.0;
      }
    }

    int bucket_length = num_pred_buckets * sizeof(int64_t);

    // will stat auc unlimited.
    if (slide_steps == 0) {
      for (int slide = 0; slide < num_pred_buckets; ++slide) {
        origin_stat_pos[slide] += (*stat_pos)[slide];
        origin_stat_neg[slide] += (*stat_neg)[slide];
      }

      *stat_pos = origin_stat_pos;
      *stat_neg = origin_stat_neg;

    } else {
      for (int slide = 1; slide < slide_steps; ++slide) {
        int dst_idx = (slide - 1) * num_pred_buckets;
        int src_inx = slide * num_pred_buckets;
        std::memcpy(origin_stat_pos + dst_idx, origin_stat_pos + src_inx,
                    bucket_length);
        std::memcpy(origin_stat_neg + dst_idx, origin_stat_neg + src_inx,
                    bucket_length);
      }

      std::memcpy(origin_stat_pos + (slide_steps - 1) * num_pred_buckets,
                  *stat_pos, bucket_length);
      std::memcpy(origin_stat_neg + (slide_steps - 1) * num_pred_buckets,
                  *stat_neg, bucket_length);

      std::memset(*stat_pos, 0, bucket_length);
      std::memset(*stat_neg, 0, bucket_length);

      for (int slide = 0; slide < num_pred_buckets; ++slide) {
        int stat_pos_steps = 0;
        int stat_neg_steps = 0;
        for (int step = 0; step < slide_steps; ++step) {
          stat_pos_steps += origin_stat_pos[slide + step * num_pred_buckets];
          stat_neg_steps += origin_stat_neg[slide + step * num_pred_buckets];
        }
        (*stat_pos)[slide] += stat_pos_steps;
        (*stat_neg)[slide] += stat_neg_steps;
      }
    }
  }

  inline static void calcAuc(const framework::ExecutionContext &ctx,
                             int64_t *stat_pos, int64_t *stat_neg,
                             int num_thresholds,
                             framework::Tensor *auc_tensor) {
    auto *auc = auc_tensor->mutable_data<double>(ctx.GetPlace());

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
