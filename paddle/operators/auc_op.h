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
#include <algorithm>
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class AccuracyKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* inference = ctx.Input<Tensor>("Inference");
    auto* inference_prob = ctx.Input<Tensor>("InferenceProb");
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

    const int* inference_data = inference->data<int>();
    const T* inference_prob_data = inference->data<T>();
    const T* label_data = label->data<T>();

    size_t num_samples = inference->dims()[0];
    size_t class_dim = inference->dims()[1];

    // create local tensor for storing the curve: TP, FN, TN, FP
    // TODO(typhoonzero): put these tensors in Scope
    // TODO(typhoonzero): use op to caculate these values.
    Tensor true_positive, false_positeve, true_negative, false_negative;

    true_positive.Resize({num_thresholds});
    false_negative.Resize({num_thresholds});
    true_negative.Resize({num_thresholds});
    false_positive.Resize({num_thresholds});

    int* tp_data = true_positive.mutable_data<int>();
    int* fn_data = false_negative.mutable_data<int>();
    int* tn_data = true_negative.mutable_data<int>();
    int* fp_data = false_positive.mutable_data<int>();

    for (auto thresh = thresholds_list.begin(); thresh != thresholds_list.end();
         thresh++) {
      size_t idx_thresh = thresh - thresholds_list.begin();
      // caculate TP, FN, TN, FP for current thresh
      int tp, fn, tn, fp = 0;
      for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < class_dim; j++) {
          if (inference_data[i * class_dim + j] == label_data[i]) {
            if (inference_prob_data[i * class_dim + j] >= (*thresh)) {
              tp++;
            } else {
              tn++;
            }
          } else {
            if (inference_prob_data[i * class_dim + j] >= (*thresh)) {
              fp++;
            } else {
              fn++;
            }
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
    float* tp_rate_data = tp_rate.mutable_data<float>();
    float* fp_rate_data = fp_rate.mutable_data<float>();
    float* rec_rate_data = rec_rate.mutable_data<float>();
    for (int i = 0; i < num_thresholds; i++) {
        tp_rate_data[i] = ((float)tp_data[i + epsilon) / (tp_data[i] + fn_data[i] + epsilon);
        fp_rate_data[i] =
            (float)fp_data[i] / (fp_data[i] + tn_data[i] + epsilon);
        rec_rate_data[i] =
            ((float)tp_data[i] + epsilon) / (tp_data[i] + fp_data[i] + epsilon);
    }

    if (curve == "ROC") {
      for (int i = 1; i < num_thresholds; i++) {
        auto dx = fp_rate_data[i] - fp_rate_data[i - 1];
        auto y = (tp_rate_data[i] + tp_rate_data[i - 1]) / 2.0f;
        *auc_data = *auc_data + dx * y;
      }
    } else if (curve = "PR") {
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
