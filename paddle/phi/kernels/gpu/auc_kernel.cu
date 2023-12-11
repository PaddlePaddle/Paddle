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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

__global__ void ClearObsoleteDataKernel(int64_t *pos,
                                        int64_t *neg,
                                        const int bucket_length,
                                        const int slide_steps) {
  int cur_step_index =
      static_cast<int>(pos[(slide_steps + 1) * bucket_length]) % slide_steps;
  int cur_step_begin = cur_step_index * bucket_length;
  int sum_step_begin = slide_steps * bucket_length;
  CUDA_KERNEL_LOOP(i, bucket_length) {
    pos[sum_step_begin + i] -= pos[cur_step_begin + i];
    neg[sum_step_begin + i] -= neg[cur_step_begin + i];
    pos[cur_step_begin + i] = neg[cur_step_begin + i] = 0;
  }
}

__global__ void UpdateSumDataKernel(int64_t *pos,
                                    int64_t *neg,
                                    const int bucket_length,
                                    const int slide_steps) {
  int cur_step_index =
      static_cast<int>(pos[(slide_steps + 1) * bucket_length]) % slide_steps;
  int cur_step_begin = cur_step_index * bucket_length;
  int sum_step_begin = slide_steps * bucket_length;
  CUDA_KERNEL_LOOP(i, bucket_length) {
    pos[sum_step_begin + i] += pos[cur_step_begin + i];
    neg[sum_step_begin + i] += neg[cur_step_begin + i];
  }
}

template <typename T>
__global__ void AddDataKernel(const int64_t *label_data,
                              const T *pred_data,
                              const int inference_width,
                              const int num_thresholds,
                              int64_t *pos,
                              int64_t *neg,
                              const int numel,
                              const int slide_steps) {
  int cur_step_begin = 0;
  if (slide_steps > 0) {
    int cur_step_index =
        static_cast<int>(pos[(slide_steps + 1) * (1 + num_thresholds)]) %
        slide_steps;
    cur_step_begin = cur_step_index * (1 + num_thresholds);
  }
  CUDA_KERNEL_LOOP(i, numel) {
    auto predict_data = pred_data[i * inference_width + (inference_width - 1)];
    PADDLE_ENFORCE(predict_data <= 1, "The predict data must less or equal 1.");
    PADDLE_ENFORCE(predict_data >= 0,
                   "The predict data must gather or equal 0.");
    uint32_t binIdx = static_cast<uint32_t>(predict_data * num_thresholds);
    if (label_data[i]) {
      phi::CudaAtomicAdd(pos + cur_step_begin + binIdx, 1);
    } else {
      phi::CudaAtomicAdd(neg + cur_step_begin + binIdx, 1);
    }
  }
}

__global__ void CalcAucKernel(int64_t *stat_pos,
                              int64_t *stat_neg,
                              int num_thresholds,
                              double *auc,
                              bool need_add_batch_num) {
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
    *auc += (totNeg - totNegPrev) * (totPos + totPosPrev) / 2.0;
    --idx;
  }

  if (totPos > 0.0 && totNeg > 0.0) {
    *auc = *auc / totPos / totNeg;
  }
  if (need_add_batch_num) {
    stat_pos[num_thresholds + 1] += 1;
    stat_neg[num_thresholds + 1] += 1;
  }
}

inline static double trapezoidArea(double X1, double X2, double Y1, double Y2) {
  return (X1 > X2 ? (X1 - X2) : (X2 - X1)) * (Y1 + Y2) / 2.0;
}

template <typename T, typename Context>
void statAuc(const Context &dev_ctx,
             const DenseTensor &label,
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
    AddDataKernel<<<(batch_size + PADDLE_CUDA_NUM_THREADS - 1) /
                        PADDLE_CUDA_NUM_THREADS,
                    PADDLE_CUDA_NUM_THREADS,
                    0,
                    dev_ctx.stream()>>>(label_data,
                                        inference_data,
                                        inference_width,
                                        num_thresholds,
                                        origin_stat_pos,
                                        origin_stat_neg,
                                        batch_size,
                                        slide_steps);
    return;
  }
  // the last number of origin_stat_pos store the index should be used in
  // current step
  int cur_step_index =
      static_cast<int>(origin_stat_pos[(slide_steps + 1) * bucket_length]) %
      slide_steps;
  int cur_step_begin = cur_step_index * bucket_length;
  int sum_step_begin = slide_steps * bucket_length;

  ClearObsoleteDataKernel<<<(bucket_length + PADDLE_CUDA_NUM_THREADS - 1) /
                                PADDLE_CUDA_NUM_THREADS,
                            PADDLE_CUDA_NUM_THREADS,
                            0,
                            dev_ctx.stream()>>>(
      origin_stat_pos, origin_stat_neg, bucket_length, slide_steps);

  AddDataKernel<<<(batch_size + PADDLE_CUDA_NUM_THREADS - 1) /
                      PADDLE_CUDA_NUM_THREADS,
                  PADDLE_CUDA_NUM_THREADS,
                  0,
                  dev_ctx.stream()>>>(label_data,
                                      inference_data,
                                      inference_width,
                                      num_thresholds,
                                      origin_stat_pos,
                                      origin_stat_neg,
                                      batch_size,
                                      slide_steps);
  if (!is_fake_data) {
    UpdateSumDataKernel<<<(bucket_length + PADDLE_CUDA_NUM_THREADS - 1) /
                              PADDLE_CUDA_NUM_THREADS,
                          PADDLE_CUDA_NUM_THREADS,
                          0,
                          dev_ctx.stream()>>>(
        origin_stat_pos, origin_stat_neg, bucket_length, slide_steps);
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

  auto *stat_pos_in_tensor = &stat_pos;
  auto *stat_neg_in_tensor = &stat_neg;
  auto *pos_in_data = stat_pos.data<int64_t>();
  auto *neg_in_data = stat_neg.data<int64_t>();
  bool is_fake_data = false;
  if (ins_tag_weight.get_ptr() != nullptr) {
    const auto *ins_tag_weight_data = ins_tag_weight->data<float>();
    if (ins_tag_weight_data[0] == 0) {
      is_fake_data = true;
    }
  }

#ifdef PADDLE_WITH_CUDA
  if (stat_pos_in_tensor != stat_pos_out) {
    cudaMemcpyAsync(
        origin_stat_pos,
        pos_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t),
        cudaMemcpyDeviceToDevice,
        dev_ctx.stream());
  }
  if (stat_neg_in_tensor != stat_neg_out) {
    cudaMemcpyAsync(
        origin_stat_neg,
        neg_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t),
        cudaMemcpyDeviceToDevice,
        dev_ctx.stream());
  }
#else
  if (stat_pos_in_tensor != stat_pos_out) {
    hipMemcpy(
        origin_stat_pos,
        pos_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t),
        hipMemcpyDeviceToDevice);
  }
  if (stat_neg_in_tensor != stat_neg_out) {
    hipMemcpy(
        origin_stat_neg,
        neg_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t),
        hipMemcpyDeviceToDevice);
  }
#endif

  // when calculate global_auc && is fake data, just do nothing
  if (slide_steps == 0 && is_fake_data) {
    return;
  }

  statAuc<T, Context>(dev_ctx,
                      label,
                      input,
                      num_thresholds,
                      slide_steps,
                      origin_stat_pos,
                      origin_stat_neg,
                      is_fake_data);
  int sum_offset = slide_steps * (num_thresholds + 1);
  CalcAucKernel<<<1, 1, 0, dev_ctx.stream()>>>(origin_stat_pos + sum_offset,
                                               origin_stat_neg + sum_offset,
                                               num_thresholds,
                                               auc_value,
                                               slide_steps > 0);
}

}  // namespace phi

PD_REGISTER_KERNEL(auc, GPU, ALL_LAYOUT, phi::AucKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT64);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT64);
}
