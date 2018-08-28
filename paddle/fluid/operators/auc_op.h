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

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class AucKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* predict = ctx.Input<Tensor>("Predict");
    auto* label = ctx.Input<Tensor>("Label");
    auto* auc = ctx.Output<Tensor>("AUC");
    // Only use output var for now, make sure it's persistable and
    // not cleaned up for each batch.
    auto* true_positive = ctx.Output<Tensor>("TPOut");
    auto* false_positive = ctx.Output<Tensor>("FPOut");
    auto* true_negative = ctx.Output<Tensor>("TNOut");
    auto* false_negative = ctx.Output<Tensor>("FNOut");

    auto* auc_data = auc->mutable_data<double>(ctx.GetPlace());

    std::string curve = ctx.Attr<std::string>("curve");
    int num_thresholds = ctx.Attr<int>("num_thresholds");
    std::vector<double> thresholds_list;
    thresholds_list.reserve(num_thresholds);
    for (int i = 1; i < num_thresholds - 1; i++) {
      thresholds_list[i] = static_cast<double>(i) / (num_thresholds - 1);
    }
    const double kEpsilon = 1e-7;
    thresholds_list[0] = 0.0f - kEpsilon;
    thresholds_list[num_thresholds - 1] = 1.0f + kEpsilon;

    size_t batch_size = predict->dims()[0];
    size_t inference_width = predict->dims()[1];

    const T* inference_data = predict->data<T>();
    const auto* label_data = label->data<int64_t>();

    auto* tp_data = true_positive->mutable_data<int64_t>(ctx.GetPlace());
    auto* fn_data = false_negative->mutable_data<int64_t>(ctx.GetPlace());
    auto* tn_data = true_negative->mutable_data<int64_t>(ctx.GetPlace());
    auto* fp_data = false_positive->mutable_data<int64_t>(ctx.GetPlace());

    for (int idx_thresh = 0; idx_thresh < num_thresholds; idx_thresh++) {
      // calculate TP, FN, TN, FP for current thresh
      int64_t tp = 0, fn = 0, tn = 0, fp = 0;
      for (size_t i = 0; i < batch_size; i++) {
        // NOTE: label_data used as bool, labels > 0 will be treated as true.
        if (label_data[i]) {
          if (inference_data[i * inference_width + 1] >=
              (thresholds_list[idx_thresh])) {
            tp++;
          } else {
            fn++;
          }
        } else {
          if (inference_data[i * inference_width + 1] >=
              (thresholds_list[idx_thresh])) {
            fp++;
          } else {
            tn++;
          }
        }
      }
      // store rates
      tp_data[idx_thresh] += tp;
      fn_data[idx_thresh] += fn;
      tn_data[idx_thresh] += tn;
      fp_data[idx_thresh] += fp;
    }
    // epsilon to avoid divide by zero.
    double epsilon = 1e-6;
    // Riemann sum to caculate auc.
    Tensor tp_rate, fp_rate, rec_rate;
    tp_rate.Resize({num_thresholds});
    fp_rate.Resize({num_thresholds});
    rec_rate.Resize({num_thresholds});
    auto* tp_rate_data = tp_rate.mutable_data<double>(ctx.GetPlace());
    auto* fp_rate_data = fp_rate.mutable_data<double>(ctx.GetPlace());
    auto* rec_rate_data = rec_rate.mutable_data<double>(ctx.GetPlace());
    for (int i = 0; i < num_thresholds; i++) {
      tp_rate_data[i] = (static_cast<double>(tp_data[i]) + epsilon) /
                        (tp_data[i] + fn_data[i] + epsilon);
      fp_rate_data[i] =
          static_cast<double>(fp_data[i]) / (fp_data[i] + tn_data[i] + epsilon);
      rec_rate_data[i] = (static_cast<double>(tp_data[i]) + epsilon) /
                         (tp_data[i] + fp_data[i] + epsilon);
    }
    *auc_data = 0.0f;
    if (curve == "ROC") {
      for (int i = 0; i < num_thresholds - 1; i++) {
        auto dx = fp_rate_data[i] - fp_rate_data[i + 1];
        auto y = (tp_rate_data[i] + tp_rate_data[i + 1]) / 2.0f;
        *auc_data = *auc_data + dx * y;
      }
    } else if (curve == "PR") {
      for (int i = 1; i < num_thresholds; i++) {
        auto dx = tp_rate_data[i] - tp_rate_data[i - 1];
        auto y = (rec_rate_data[i] + rec_rate_data[i - 1]) / 2.0f;
        *auc_data = *auc_data + dx * y;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
