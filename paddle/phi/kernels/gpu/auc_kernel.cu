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

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include <cub/cub.cuh>

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

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
                              const int slide_steps,
                              bool ignore_illegal_label) {
  int cur_step_begin = 0;
  if (slide_steps > 0) {
    int cur_step_index =
        static_cast<int>(pos[(slide_steps + 1) * (1 + num_thresholds)]) %
        slide_steps;
    cur_step_begin = cur_step_index * (1 + num_thresholds);
  }
  CUDA_KERNEL_LOOP(i, numel) {
    int64_t label = label_data[i];
    if (ignore_illegal_label) {
      if (label != 0 && label != 1) {
        continue;
      }
    }
    auto predict_data = pred_data[i * inference_width + (inference_width - 1)];
    PADDLE_ENFORCE(predict_data <= 1, "The predict data(%f) must less or equal 1.", predict_data);
    PADDLE_ENFORCE(predict_data >= 0,
                   "The predict data(%f) must gather or equal 0.", predict_data);
    uint32_t binIdx = static_cast<uint32_t>(predict_data * num_thresholds);
    if (label) {
      paddle::platform::CudaAtomicAdd(pos + cur_step_begin + binIdx, 1);
    } else {
      paddle::platform::CudaAtomicAdd(neg + cur_step_begin + binIdx, 1);
    }
  }
}

template <int BLOCKDIM>
__global__ void CalcAucKernel(int64_t *stat_pos,
                              int64_t *stat_neg,
                              int num_thresholds,
                              double *auc,
                              bool need_add_batch_num) {
  typedef cub::BlockScan<int64_t, BLOCKDIM> Int64BlockScan;
  __shared__ typename Int64BlockScan::TempStorage int64_scan_storage;

  typedef cub::BlockReduce<int64_t, BLOCKDIM> Int64BlockReduce;
  __shared__ typename Int64BlockReduce::TempStorage int64_reduce_storage;

  typedef cub::BlockReduce<double, BLOCKDIM> DoubleBlockReduce;
  __shared__ typename DoubleBlockReduce::TempStorage double_reduce_storage;

  int64_t total_pos_num_local = 0; // thread_local_num
  int64_t total_neg_num = 0; // global_num

  double area_local = 0.0;
  int block_begin_idx = 0;
  for (; block_begin_idx < num_thresholds; block_begin_idx += BLOCKDIM) {
    int idx = block_begin_idx + threadIdx.x;
    int64_t pos_num = 0;
    int64_t neg_num = 0;
    if (idx <= num_thresholds) {
      pos_num = stat_pos[idx];
      neg_num = stat_neg[idx];
    }
    total_pos_num_local += pos_num;

    int64_t block_aggregate = 0;
    int64_t neg_prefix_sum = 0;
    __syncthreads();
    Int64BlockScan(int64_scan_storage).ExclusiveSum(neg_num, neg_prefix_sum, block_aggregate);
    
    neg_prefix_sum += total_neg_num;
    total_neg_num += block_aggregate;
    area_local += static_cast<double>(pos_num) * (neg_prefix_sum + neg_prefix_sum + neg_num);
  }
  
  int64_t total_pos_num = Int64BlockReduce(int64_reduce_storage).Sum(total_pos_num_local);
  double area = DoubleBlockReduce(double_reduce_storage).Sum(area_local);
  
  if (threadIdx.x == 0) {
    if (block_begin_idx == num_thresholds) {
      // for num_thresholds % BLOCKDIM == 0
      int64_t pos_num = stat_pos[num_thresholds];
      int64_t neg_num = stat_neg[num_thresholds];
      area += static_cast<double>(pos_num) * (total_neg_num + total_neg_num + neg_num);
      total_pos_num += pos_num;
      total_neg_num += neg_num;
    }
    if (total_pos_num == 0 || total_neg_num == 0) {
      *auc = 0.0;
    } else {
      *auc = area / total_pos_num / total_neg_num / 2.0;
    }
    if (need_add_batch_num) {
      stat_pos[num_thresholds + 1] += 1;
      stat_neg[num_thresholds + 1] += 1;
    }
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
             bool ignore_illegal_label,
             int64_t *origin_stat_pos,
             int64_t *origin_stat_neg) {
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
                                        slide_steps,
                                        ignore_illegal_label);
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
                                      slide_steps,
                                      ignore_illegal_label);
  UpdateSumDataKernel<<<(bucket_length + PADDLE_CUDA_NUM_THREADS - 1) /
                            PADDLE_CUDA_NUM_THREADS,
                        PADDLE_CUDA_NUM_THREADS,
                        0,
                        dev_ctx.stream()>>>(
      origin_stat_pos, origin_stat_neg, bucket_length, slide_steps);
}

template <typename T, typename Context>
void AucKernel(const Context &dev_ctx,
               const DenseTensor &input,
               const DenseTensor &label,
               const DenseTensor &stat_pos,
               const DenseTensor &stat_neg,
               const std::string &curve,
               int num_thresholds,
               int slide_steps,
               bool ignore_illegal_label,
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
  auto stream = dev_ctx.stream();
#ifdef PADDLE_WITH_CUDA
  if (stat_pos_in_tensor != stat_pos_out) {
    cudaMemcpyAsync(
        origin_stat_pos,
        pos_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t),
        cudaMemcpyDeviceToDevice, stream);
  }
  if (stat_neg_in_tensor != stat_neg_out) {
    cudaMemcpyAsync(
        origin_stat_neg,
        neg_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t),
        cudaMemcpyDeviceToDevice, stream);
  }
#else
  if (stat_pos_in_tensor != stat_pos_out) {
    hipMemcpyAsync(
        origin_stat_pos,
        pos_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t),
        hipMemcpyDeviceToDevice, stream);
  }
  if (stat_neg_in_tensor != stat_neg_out) {
    hipMemcpyAsync(
        origin_stat_neg,
        neg_in_data,
        ((1 + slide_steps) * (num_thresholds + 1) + (slide_steps > 0 ? 1 : 0)) *
            sizeof(int64_t),
        hipMemcpyDeviceToDevice, stream);
  }
#endif

  statAuc<T, Context>(dev_ctx,
                      label,
                      input,
                      num_thresholds,
                      slide_steps,
                      ignore_illegal_label,
                      origin_stat_pos,
                      origin_stat_neg);
  int sum_offset = slide_steps * (num_thresholds + 1);
  CalcAucKernel<512><<<1, 512, 0, dev_ctx.stream()>>>(origin_stat_pos + sum_offset,
                                               origin_stat_neg + sum_offset,
                                               num_thresholds,
                                               auc_value,
                                               slide_steps > 0);
}

}  // namespace phi

PD_REGISTER_KERNEL(auc, GPU, ALL_LAYOUT, phi::AucKernel, float) {}
