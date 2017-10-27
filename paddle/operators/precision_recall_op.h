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
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

enum StateVariable { TP = 0, FP, TN, FN };

template <typename Place, typename T>
class PrecisionRecallKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in0 = ctx.Input<Tensor>("Predictions");
    auto* in1 = ctx.Input<Tensor>("Labels");
    auto* in2 = ctx.Input<Tensor>("Weights");
    auto* in3 = ctx.Input<Tensor>("StatesInfo");
    auto* out0 = ctx.Output<Tensor>("BatchMetrics");
    auto* out1 = ctx.Output<Tensor>("AccumMetrics");
    auto* out2 = ctx.Output<Tensor>("AccumStatesInfo");

    const T* predictions_data = in0->data<T>();
    const int* labels_data = in1->data<int>();
    const T* weights_data = in2 ? in2->data<T>() : nullptr;
    const T* states_data = in3 ? in3->data<T>() : nullptr;
    double* batch_metrics_data = out0->mutable_data<double>(ctx.GetPlace());
    double* accum_metrics_data = out1->mutable_data<double>(ctx.GetPlace());
    out2->mutable_data<T>(ctx.GetPlace());
    auto accum_states = EigenMatrix<T>::From(*out2);
    accum_states.setZero();
    T* accum_states_data = out2->data<T>();

    size_t sample_num = in0->dims()[0];
    size_t class_dim = in0->dims()[1];
    size_t state_var_num = 4;  // TP FP TN FN

    // get states info for current batch
    for (size_t i = 0; i < sample_num; ++i) {
      size_t max_idx = 0;
      T max_val = predictions_data[i * class_dim];
      for (size_t j = 1; j < class_dim; ++j) {
        if (max_val < predictions_data[i * class_dim + j]) {
          max_idx = j;
          max_val = predictions_data[i * class_dim + j];
        }
      }

      T w = weights_data ? weights_data[i] : 1.0;
      if (max_idx == labels_data[i]) {
        accum_states_data[max_idx * state_var_num + TP] += w;
        for (size_t j = 0; j < class_dim; ++j) {
          accum_states_data[j * state_var_num + TN] += w;
        }
        accum_states_data[max_idx * state_var_num + TN] -= w;
      } else {
        accum_states_data[labels_data[i] * state_var_num + FN] += w;
        accum_states_data[max_idx * state_var_num + FP] += w;
        for (size_t j = 0; j < class_dim; ++j) {
          accum_states_data[j * state_var_num + TN] += w;
        }
        accum_states_data[max_idx * state_var_num + TN] -= w;
        accum_states_data[labels_data[i] * state_var_num + TN] -= w;
      }
    }

    ComputeMetrics(accum_states_data, batch_metrics_data, state_var_num,
                   class_dim);

    if (states_data) {
      for (size_t i = 0; i < class_dim; ++i) {
        for (size_t j = 0; j < state_var_num; ++j) {
          size_t idx = i * state_var_num + j;
          accum_states_data[idx] += states_data[idx];
        }
      }
    }

    ComputeMetrics(accum_states_data, accum_metrics_data, state_var_num,
                   class_dim);
  }

  // expose to be reused
  static inline T CalcPrecision(T tp_count, T fp_count) {
    if (tp_count > 0.0 || fp_count > 0.0) {
      return tp_count / (tp_count + fp_count);
    }
    return 1.0;
  }

  static inline T CalcRecall(T tp_count, T fn_count) {
    if (tp_count > 0.0 || fn_count > 0.0) {
      return tp_count / (tp_count + fn_count);
    }
    return 1.0;
  }

  static inline T CalcF1Score(T precision, T recall) {
    if (precision > 0.0 || recall > 0.0) {
      return 2 * precision * recall / (precision + recall);
    }
    return 0.0;
  }

 protected:
  void ComputeMetrics(const T* states_data, double* metrics_data,
                      size_t state_var_num, size_t class_dim) const {
    T total_tp_count = 0;
    T total_fp_count = 0;
    T total_fn_count = 0;
    T macro_avg_precision = 0.0;
    T macro_avg_recall = 0.0;

    for (size_t i = 0; i < class_dim; ++i) {
      T tp_count = states_data[i * state_var_num + TP];
      T fp_count = states_data[i * state_var_num + FP];
      T fn_count = states_data[i * state_var_num + FN];
      total_tp_count += tp_count;
      total_fp_count += fp_count;
      total_fn_count += fn_count;
      macro_avg_precision += CalcPrecision(tp_count, fp_count);
      macro_avg_recall += CalcRecall(tp_count, fn_count);
    }
    macro_avg_precision /= class_dim;
    macro_avg_recall /= class_dim;
    T macro_f1_score = CalcF1Score(macro_avg_precision, macro_avg_recall);

    T micro_avg_precision = CalcPrecision(total_tp_count, total_fp_count);
    T micro_avg_recall = CalcRecall(total_tp_count, total_fn_count);
    T micro_f1_score = CalcF1Score(micro_avg_precision, micro_avg_recall);

    // fill metrics data
    metrics_data[0] = macro_avg_precision;
    metrics_data[1] = macro_avg_recall;
    metrics_data[2] = macro_f1_score;
    metrics_data[3] = micro_avg_precision;
    metrics_data[4] = micro_avg_recall;
    metrics_data[5] = micro_f1_score;
  }
};

}  // namespace operators
}  // namespace paddle
