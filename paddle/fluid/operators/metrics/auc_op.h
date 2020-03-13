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

    int num_thresholds = ctx.Attr<int>("num_thresholds");
    int slide_steps = ctx.Attr<int>("slide_steps");

    // Only use output var for now, make sure it's persistable and
    // not cleaned up for each batch.
    auto *auc_tensor = ctx.Output<Tensor>("AUC");
    auto *stat_pos = ctx.Output<Tensor>("StatPosOut");
    auto *stat_neg = ctx.Output<Tensor>("StatNegOut");

    auto *origin_stat_pos = stat_pos->mutable_data<int64_t>(ctx.GetPlace());
    auto *origin_stat_neg = stat_neg->mutable_data<int64_t>(ctx.GetPlace());
    auto *auc_value = auc_tensor->mutable_data<double>(ctx.GetPlace());

    // Just for pass UT, since UT's input & output connot be set same var
    auto *stat_pos_in_tensor = ctx.Input<Tensor>("StatPos");
    auto *pos_in_data = stat_pos_in_tensor->data<int64_t>();
    auto *stat_neg_in_tensor = ctx.Input<Tensor>("StatNeg");
    auto *neg_in_data = stat_neg_in_tensor->data<int64_t>();
    if (stat_pos_in_tensor != stat_pos) {
      memcpy(origin_stat_pos, pos_in_data,
             ((1 + slide_steps) * (num_thresholds + 1) +
              (slide_steps > 0 ? 1 : 0)) *
                 sizeof(int64_t));
    }
    if (stat_neg_in_tensor != stat_neg) {
      memcpy(origin_stat_neg, neg_in_data,
             ((1 + slide_steps) * (num_thresholds + 1) +
              (slide_steps > 0 ? 1 : 0)) *
                 sizeof(int64_t));
    }
    statAuc(label, predict, num_thresholds, slide_steps, origin_stat_pos,
            origin_stat_neg);

    int sum_offset = slide_steps * (num_thresholds + 1);
    calcAuc(origin_stat_pos + sum_offset, origin_stat_neg + sum_offset,
            num_thresholds, auc_value);
    if (slide_steps) {
      origin_stat_pos[(slide_steps + 1) * (num_thresholds + 1)] += 1;
      origin_stat_neg[(slide_steps + 1) * (num_thresholds + 1)] += 1;
    }
  }

 private:
  inline static double trapezoidArea(double X1, double X2, double Y1,
                                     double Y2) {
    return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
  }

  inline static void statAuc(const framework::Tensor *label,
                             const framework::Tensor *predict,
                             const int num_thresholds, const int slide_steps,
                             int64_t *origin_stat_pos,
                             int64_t *origin_stat_neg) {
    size_t batch_size = predict->dims()[0];
    size_t inference_width = predict->dims()[1];
    const T *inference_data = predict->data<T>();
    const auto *label_data = label->data<int64_t>();
    const int bucket_length = num_thresholds + 1;
    if (slide_steps == 0) {
      for (size_t i = 0; i < batch_size; i++) {
        // if predict_data[i] has dim of 2, then predict_data[i][1] is pos prob
        // if predict_data[i] has dim of 1, then predict_data[i][0] is pos prob
        auto predict_data =
            inference_data[i * inference_width + (inference_width - 1)];
        PADDLE_ENFORCE_LE(predict_data, 1,
                          platform::errors::PreconditionNotMet(
                              "The predict data must less or equal 1."));
        PADDLE_ENFORCE_GE(predict_data, 0,
                          platform::errors::PreconditionNotMet(
                              "The predict data must gather or equal 0."));

        uint32_t binIdx = static_cast<uint32_t>(predict_data * num_thresholds);
        if (label_data[i] > 0) {
          origin_stat_pos[binIdx] += 1;
        } else if (label_data[i] == 0) {
          origin_stat_neg[binIdx] += 1;
        }
      }
      return;
    }
    // the last number of origin_stat_pos store the index should be used in
    // current step
    int cur_step_index =
        static_cast<int>(origin_stat_pos[(slide_steps + 1) * bucket_length]) %
        slide_steps;
    int cur_step_begin = cur_step_index * bucket_length;
    int sum_step_begin = slide_steps * bucket_length;
    for (int i = 0; i < bucket_length; ++i) {
      origin_stat_pos[sum_step_begin + i] -=
          origin_stat_pos[cur_step_begin + i];
      origin_stat_neg[sum_step_begin + i] -=
          origin_stat_neg[cur_step_begin + i];
    }

    std::memset(origin_stat_pos + cur_step_begin, 0,
                bucket_length * sizeof(int64_t));
    std::memset(origin_stat_neg + cur_step_begin, 0,
                bucket_length * sizeof(int64_t));

    for (size_t i = 0; i < batch_size; i++) {
      // if predict_data[i] has dim of 2, then predict_data[i][1] is pos prob
      // if predict_data[i] has dim of 1, then predict_data[i][0] is pos prob
      auto predict_data =
          inference_data[i * inference_width + (inference_width - 1)];
      PADDLE_ENFORCE_LE(predict_data, 1,
                        platform::errors::PreconditionNotMet(
                            "The predict data must less or equal 1."));
      PADDLE_ENFORCE_GE(predict_data, 0,
                        platform::errors::PreconditionNotMet(
                            "The predict data must gather or equal 0."));

      uint32_t binIdx = static_cast<uint32_t>(predict_data * num_thresholds);
      if (label_data[i] > 0) {
        origin_stat_pos[cur_step_begin + binIdx] += 1;
      } else if (label_data[i] == 0) {
        origin_stat_neg[cur_step_begin + binIdx] += 1;
      }
    }
    for (int i = 0; i < bucket_length; ++i) {
      origin_stat_pos[sum_step_begin + i] +=
          origin_stat_pos[cur_step_begin + i];
      origin_stat_neg[sum_step_begin + i] +=
          origin_stat_neg[cur_step_begin + i];
    }
  }

  inline static void calcAuc(const int64_t *stat_pos, const int64_t *stat_neg,
                             int num_thresholds, double *auc) {
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
