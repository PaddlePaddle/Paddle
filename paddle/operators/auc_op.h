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

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename Place, typename T>
class AucKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* inference = ctx.Input<Tensor>("Inference");
    auto* label = ctx.Input<Tensor>("Label");
    auto* auc = ctx.Output<Tensor>("AUC");

    float* auc_data = auc->mutable_data<float>(ctx.GetPlace());

    std::string curve = ctx.Attr<std::string>("curve");
    int num_thresholds = ctx.Attr<int>("num_thresholds");
    std::vector<float> thresholds_list;
    thresholds_list.reserve(num_thresholds);
    for (int i = 1; i < num_thresholds - 1; i++) {
      thresholds_list[i] = (float)i / (num_thresholds - 1);
    }
    const float kEpsilon = 1e-7;
    thresholds_list[0] = 0.0f - kEpsilon;
    thresholds_list[num_thresholds - 1] = 1.0f + kEpsilon;

    size_t num_samples = inference->numel();

    const T* inference_data = inference->data<T>();
    Tensor label_casted;
    label_casted.Resize(label->dims());
    bool* label_casted_data = label_casted.mutable_data<bool>(ctx.GetPlace());

    const int* label_data = label->data<int>();
    // cast label_data to bool
    for (size_t i = 0; i < num_samples; i++) {
      label_casted_data[i] = static_cast<bool>(label_data[i]);
    }

    // Create local tensor for storing the curve: TP, FN, TN, FP
    // TODO(typhoonzero): use eigen op to caculate these values.
    Tensor true_positive, false_positive, true_negative, false_negative;

    true_positive.Resize({num_thresholds});
    false_negative.Resize({num_thresholds});
    true_negative.Resize({num_thresholds});
    false_positive.Resize({num_thresholds});

    int* tp_data = true_positive.mutable_data<int>(ctx.GetPlace());
    int* fn_data = false_negative.mutable_data<int>(ctx.GetPlace());
    int* tn_data = true_negative.mutable_data<int>(ctx.GetPlace());
    int* fp_data = false_positive.mutable_data<int>(ctx.GetPlace());

    for (int idx_thresh = 0; idx_thresh < num_thresholds; idx_thresh++) {
      // caculate TP, FN, TN, FP for current thresh
      int tp = 0, fn = 0, tn = 0, fp = 0;
      for (size_t i = 0; i < num_samples; i++) {
        if (label_casted_data[i]) {
          if (inference_data[i] >= (thresholds_list[idx_thresh])) {
            tp++;
          } else {
            fn++;
          }
        } else {
          if (inference_data[i] >= (thresholds_list[idx_thresh])) {
            fp++;
          } else {
            tn++;
          }
        }
      }
      // store rates
      tp_data[idx_thresh] = tp;
      fn_data[idx_thresh] = fn;
      tn_data[idx_thresh] = tn;
      fp_data[idx_thresh] = fp;
    }
    // epsilon to avoid divide by zero.
    float epsilon = 1e-6;
    // Riemann sum to caculate auc.
    Tensor tp_rate, fp_rate, rec_rate;
    tp_rate.Resize({num_thresholds});
    fp_rate.Resize({num_thresholds});
    rec_rate.Resize({num_thresholds});
    float* tp_rate_data = tp_rate.mutable_data<float>(ctx.GetPlace());
    float* fp_rate_data = fp_rate.mutable_data<float>(ctx.GetPlace());
    float* rec_rate_data = rec_rate.mutable_data<float>(ctx.GetPlace());
    for (int i = 0; i < num_thresholds; i++) {
      tp_rate_data[i] =
          ((float)tp_data[i] + epsilon) / (tp_data[i] + fn_data[i] + epsilon);
      fp_rate_data[i] = (float)fp_data[i] / (fp_data[i] + tn_data[i] + epsilon);
      rec_rate_data[i] =
          ((float)tp_data[i] + epsilon) / (tp_data[i] + fp_data[i] + epsilon);
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
