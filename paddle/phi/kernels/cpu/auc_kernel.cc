// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/auc_kernel.h"
#include <glog/logging.h>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

inline static double trapezoidArea(double X1, double X2, double Y1, double Y2) {
  return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
}

inline static size_t compute_max_bytes(int64_t *dest,
                                       const long *src,  // NOLINT
                                       const int num_thresholds,
                                       const int slide_steps) {
  return reinterpret_cast<const char *>(src + (num_thresholds + 1) *
                                                  (slide_steps + 1)) -
         reinterpret_cast<const char *>(dest);
}

template <typename T>
void statAuc(const DenseTensor &label,
             const DenseTensor &predict,
             const int num_thresholds,
             const int slide_steps,
             int64_t *origin_stat_pos,
             int64_t *origin_stat_neg,
             const bool is_fake_data) {
  size_t batch_size = predict.dims()[0];
  size_t inference_width = predict.dims()[1];
  const T *inference_data = predict.data<T>();
  const auto *label_data = label.data<int64_t>();
  const int bucket_length = num_thresholds + 1;
  if (slide_steps == 0) {
    for (size_t i = 0; i < batch_size; i++) {
      // if predict_data[i] has dim of 2, then predict_data[i][1] is pos prob
      // if predict_data[i] has dim of 1, then predict_data[i][0] is pos prob
      auto predict_data =
          inference_data[i * inference_width + (inference_width - 1)];
      PADDLE_ENFORCE_LE(predict_data,
                        1,
                        common::errors::PreconditionNotMet(
                            "The predict data must less or equal 1."));
      PADDLE_ENFORCE_GE(predict_data,
                        0,
                        common::errors::PreconditionNotMet(
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
    origin_stat_pos[sum_step_begin + i] -= origin_stat_pos[cur_step_begin + i];
    origin_stat_neg[sum_step_begin + i] -= origin_stat_neg[cur_step_begin + i];
  }

  std::memset(
      origin_stat_pos + cur_step_begin, 0, bucket_length * sizeof(int64_t));
  std::memset(
      origin_stat_neg + cur_step_begin, 0, bucket_length * sizeof(int64_t));

  for (size_t i = 0; i < batch_size; i++) {
    // if predict_data[i] has dim of 2, then predict_data[i][1] is pos prob
    // if predict_data[i] has dim of 1, then predict_data[i][0] is pos prob
    auto predict_data =
        inference_data[i * inference_width + (inference_width - 1)];
    PADDLE_ENFORCE_LE(predict_data,
                      1,
                      common::errors::PreconditionNotMet(
                          "The predict data must less or equal 1."));
    PADDLE_ENFORCE_GE(predict_data,
                      0,
                      common::errors::PreconditionNotMet(
                          "The predict data must gather or equal 0."));

    uint32_t binIdx = static_cast<uint32_t>(predict_data * num_thresholds);
    if (label_data[i] > 0) {
      origin_stat_pos[cur_step_begin + binIdx] += 1;
    } else if (label_data[i] == 0) {
      origin_stat_neg[cur_step_begin + binIdx] += 1;
    }
  }
  if (!is_fake_data) {
    for (int i = 0; i < bucket_length; ++i) {
      origin_stat_pos[sum_step_begin + i] +=
          origin_stat_pos[cur_step_begin + i];
      origin_stat_neg[sum_step_begin + i] +=
          origin_stat_neg[cur_step_begin + i];
    }
  }
}

inline static void calcAuc(const int64_t *stat_pos,
                           const int64_t *stat_neg,
                           int num_thresholds,
                           double *auc) {
  *auc = 0.0f;

  double totPos = 0.0;
  double totNeg = 0.0;
  double totPosPrev = 0.0;
  double totNegPrev = 0.0;

  int idx = num_thresholds;

  while (idx >= 0) {
    totPosPrev = totPos;
    totNegPrev = totNeg;
    totPos += static_cast<double>(stat_pos[idx]);
    totNeg += static_cast<double>(stat_neg[idx]);
    *auc += trapezoidArea(totNeg, totNegPrev, totPos, totPosPrev);
    --idx;
  }

  if (totPos > 0.0 && totNeg > 0.0) {
    *auc = *auc / totPos / totNeg;
  }
}

template <typename T, typename Context>
void AucKernel(const Context &dev_ctx,
               const DenseTensor &input,
               const DenseTensor &label,
               const DenseTensor &stat_pos,
               const DenseTensor &stat_neg,
               const paddle::optional<DenseTensor> &ins_tag_weight,
               const std::string &curve,
               int num_thresholds,
               int slide_steps,
               DenseTensor *auc,
               DenseTensor *stat_pos_out,
               DenseTensor *stat_neg_out) {
  // Only use output var for now, make sure it's persistable and
  // not cleaned up for each batch.
  auto *origin_stat_pos = dev_ctx.template Alloc<int64_t>(stat_pos_out);
  auto *origin_stat_neg = dev_ctx.template Alloc<int64_t>(stat_neg_out);
  auto *auc_value = dev_ctx.template Alloc<double>(auc);

  // Just for pass UT, since UT's input & output cannot be set same var
  auto *stat_pos_in_tensor = &stat_pos;
  auto *stat_neg_in_tensor = &stat_neg;
  auto *pos_in_data = stat_pos.data<int64_t>();
  auto *neg_in_data = stat_neg.data<int64_t>();
  bool is_fake_data = false;
  if (ins_tag_weight.get_ptr() != nullptr) {
    const auto *ins_tag_weight_data = ins_tag_weight->data<float>();
    VLOG(4) << "auc ins_tag_weight = " << ins_tag_weight_data[0];
    if (ins_tag_weight_data[0] == 0) {
      is_fake_data = true;
    }
  }
  size_t required_bytes =
      ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
      sizeof(int64_t);
  if (stat_pos_in_tensor != stat_pos_out) {
    size_t max_bytes = compute_max_bytes(
        origin_stat_pos,
        reinterpret_cast<const long *>(pos_in_data),  // NOLINT
        num_thresholds,
        slide_steps);
    PADDLE_ENFORCE_LE(required_bytes,
                      max_bytes,
                      common::errors::PreconditionNotMet(
                          "The number of bytes to be copied %d must be less "
                          "than or equal to the maximum number of bytes %d. ",
                          required_bytes,
                          max_bytes));
    memcpy(
        origin_stat_pos,
        pos_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t));
  }
  if (stat_neg_in_tensor != stat_neg_out) {
    size_t max_bytes = compute_max_bytes(
        origin_stat_neg,
        reinterpret_cast<const long *>(neg_in_data),  // NOLINT
        num_thresholds,
        slide_steps);
    PADDLE_ENFORCE_LE(required_bytes,
                      max_bytes,
                      common::errors::PreconditionNotMet(
                          "The number of bytes to be copied %d must be less "
                          "than or equal to the maximum number of bytes %d. ",
                          required_bytes,
                          max_bytes));
    memcpy(
        origin_stat_neg,
        neg_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t));
  }

  // when calculate global_auc && is fake data, just do nothing
  if (slide_steps == 0 && is_fake_data) {
    return;
  }
  statAuc<T>(label,
             input,
             num_thresholds,
             slide_steps,
             origin_stat_pos,
             origin_stat_neg,
             is_fake_data);

  int sum_offset = slide_steps * (num_thresholds + 1);
  calcAuc(origin_stat_pos + sum_offset,
          origin_stat_neg + sum_offset,
          num_thresholds,
          auc_value);
  if (slide_steps) {
    origin_stat_pos[(slide_steps + 1) * (num_thresholds + 1)] += 1;
    origin_stat_neg[(slide_steps + 1) * (num_thresholds + 1)] += 1;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(auc, CPU, ALL_LAYOUT, phi::AucKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT64);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT64);
}
