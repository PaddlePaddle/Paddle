/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "cub/cub.cuh"
#include "paddle/phi/kernels/funcs/math_cuda_utils.h"

namespace phi {

static const float HALF_FLT_MAX = 65504.F;
static const float HALF_FLT_MIN = -65504.F;
static inline size_t AlignTo16(const size_t& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

class CubKeyValueSorter {
 public:
  CubKeyValueSorter();

  explicit CubKeyValueSorter(const int num_experts);

  void update_num_experts(const int num_experts);

  size_t getWorkspaceSize(const size_t num_key_value_pairs,
                          bool descending = false);

  template <typename KeyT>
  void run(void* workspace,
           const size_t workspace_size,
           const KeyT* keys_in,
           KeyT* keys_out,
           const int* values_in,
           int* values_out,
           const size_t num_key_value_pairs,
           bool descending,
           cudaStream_t stream);

 private:
  size_t num_key_value_pairs_;
  int num_experts_;
  int num_bits_;
};

// ===== CUB Sorting things =====
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0), num_bits_(sizeof(int) * 8) {}

CubKeyValueSorter::CubKeyValueSorter(const int num_experts)
    : num_experts_(num_experts),
      num_bits_(static_cast<int>(log2(num_experts)) + 1) {}

void CubKeyValueSorter::update_num_experts(const int num_experts) {
  num_experts_ = num_experts;
  num_bits_ = static_cast<int>(log2(num_experts)) + 1;
}

size_t CubKeyValueSorter::getWorkspaceSize(const size_t num_key_value_pairs,
                                           bool descending) {
  num_key_value_pairs_ = num_key_value_pairs;
  size_t required_storage = 0;
  int* null_int = nullptr;
  if (descending) {
    cub::DeviceRadixSort::SortPairsDescending(NULL,
                                              required_storage,
                                              null_int,
                                              null_int,
                                              null_int,
                                              null_int,
                                              num_key_value_pairs,
                                              0,
                                              32);
  } else {
    cub::DeviceRadixSort::SortPairs(NULL,
                                    required_storage,
                                    null_int,
                                    null_int,
                                    null_int,
                                    null_int,
                                    num_key_value_pairs,
                                    0,
                                    num_bits_);
  }
  return required_storage;
}

template <typename KeyT>
void CubKeyValueSorter::run(void* workspace,
                            const size_t workspace_size,
                            const KeyT* keys_in,
                            KeyT* keys_out,
                            const int* values_in,
                            int* values_out,
                            const size_t num_key_value_pairs,
                            bool descending,
                            cudaStream_t stream) {
  size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs);
  size_t actual_ws_size = workspace_size;

  if (expected_ws_size > workspace_size) {
    std::stringstream err_ss;
    err_ss << "[Error][CubKeyValueSorter::run]\n";
    err_ss
        << "Error. The allocated workspace is too small to run this problem.\n";
    err_ss << "Expected workspace size of at least " << expected_ws_size
           << " but got problem size " << workspace_size << "\n";
    throw std::runtime_error(err_ss.str());
  }
  if (descending) {
    cub::DeviceRadixSort::SortPairsDescending(workspace,
                                              actual_ws_size,
                                              keys_in,
                                              keys_out,
                                              values_in,
                                              values_out,
                                              num_key_value_pairs,
                                              0,
                                              32,
                                              stream);
  } else {
    cub::DeviceRadixSort::SortPairs(workspace,
                                    actual_ws_size,
                                    keys_in,
                                    keys_out,
                                    values_in,
                                    values_out,
                                    num_key_value_pairs,
                                    0,
                                    num_bits_,
                                    stream);
  }
}

template <>
void CubKeyValueSorter::run(void* workspace,
                            const size_t workspace_size,
                            const __nv_bfloat16* keys_in,
                            __nv_bfloat16* keys_out,
                            const int* values_in,
                            int* values_out,
                            const size_t num_key_value_pairs,
                            bool descending,
                            cudaStream_t stream) {}

CubKeyValueSorter sorter_;

// --------      getWorkspaceSize      -------- //
template <typename T>
size_t getWorkspaceSize(const int num_rows,
                        const int hidden_size,
                        const int inter_size,
                        const int num_experts,
                        const int k,
                        const int batch_size,
                        const int max_seq_len) {
  const int buf_size = AlignTo16(num_experts * batch_size * k * hidden_size);
  const int interbuf_size =
      AlignTo16(num_experts * batch_size * k * inter_size);
  const int padded_experts = AlignTo16(num_experts);
  const int num_moe_inputs = AlignTo16(num_experts * num_rows);
  int padded_num_moe_inputs = num_experts * batch_size * max_seq_len;

  size_t total_ws_bytes = sizeof(int) * num_moe_inputs;   // source_rows_
  total_ws_bytes += sizeof(int) * padded_num_moe_inputs;  // padded_source_rows_
  total_ws_bytes += sizeof(T) * padded_num_moe_inputs;  // padded_expert_scales_
  total_ws_bytes += sizeof(int) * padded_num_moe_inputs;  // permuted_rows_
  total_ws_bytes += sizeof(int) * num_experts * k;        // permuted_experts_
  total_ws_bytes += buf_size * sizeof(T);                 // permuted_data_
  total_ws_bytes +=
      padded_experts * sizeof(int64_t);  // Hold total_rows_before_expert_

  total_ws_bytes += sizeof(T) * num_moe_inputs;         // attr_mask: [e, n]
  total_ws_bytes += sizeof(T) * padded_num_moe_inputs;  // sorted_softmax_output

  const int bytes_for_fc1_result = interbuf_size * sizeof(T);
  const int sorter_ws_size_bytes =
      AlignTo16(sorter_.getWorkspaceSize(num_experts * k));
  sorter_.update_num_experts(k);

  int bytes_for_intermediate_and_sorting = bytes_for_fc1_result;
  if (sorter_ws_size_bytes > bytes_for_fc1_result) {
    int remaining_bytes =
        AlignTo16(sorter_ws_size_bytes - bytes_for_fc1_result);
    bytes_for_intermediate_and_sorting += remaining_bytes;
  }

  total_ws_bytes +=
      bytes_for_intermediate_and_sorting;  // intermediate (fc1) output + cub
                                           // sorting workspace
  return total_ws_bytes;
}

// --------      initialize_expert_choice_route_kernel      -------- //
template <typename T>
__global__ void initialize_expert_choice_route_kernel(
    int* expert_for_source_row,
    int* source_row,
    int* expanded_source_row_to_expanded_dest_row,
    int64_t* total_rows_before_expert,
    T* attr_mask,
    const int cols,
    const int k,
    const int batch_size) {
  int start = cols * blockIdx.x;

  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    expert_for_source_row[start + i] = blockIdx.x;
    source_row[start + i] = start + i;
    expanded_source_row_to_expanded_dest_row[start + i] = -1;
    attr_mask[start + i] = (T)1.0f;
  }
  if (threadIdx.x == 0) {
    total_rows_before_expert[blockIdx.x] = batch_size * k * (blockIdx.x + 1);
  }
}

// --------      softmax_kernel      -------- //
template <int ITEMS_PER_THREAD, typename T>
__global__ void softmax_kernel_v4(
    T* qk_buf_,
    const T* qk_buf_src,  // shape [batch_size, seq_len]
    const T* attr_mask,   // shape [batch_size, seq_len]
    const int batch_size,
    const int seq_len) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  float data[ITEMS_PER_THREAD];
  int qk_offset;
  __shared__ float s_mean, s_max;
  float local_max = -1e20f;
  for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
    qk_offset =
        ((blockIdx.y + blockIdx.z)) * seq_len + blockDim.x * i + threadIdx.x;
    int mask_offset = (blockIdx.y) * seq_len + blockDim.x * i + threadIdx.x;

    float qk = static_cast<float>(qk_buf_src[qk_offset]);
    float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

    mask_val = (1.0f - mask_val) * -10000.0f;

    data[i] = qk + mask_val;
    local_max = fmax(local_max, data[i]);
  }

  float max_val =
      blockDim.x <= 32
          ? phi::funcs::WarpReduceMax<float>(local_max, 0xFFFFFFFF)
          : phi::funcs::BlockReduceMax<float>(local_max, 0xFFFFFFFF);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();

  float local_sum = 0;
  for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
    data[i] = __expf(data[i] - s_max);
    local_sum += data[i];
  }
  float sum_val =
      blockDim.x <= 32
          ? phi::funcs::WarpReduceSum<float>(local_sum, 0xFFFFFFFF)
          : phi::funcs::BlockReduceSum<float>(local_sum, 0xFFFFFFFF);
  if (threadIdx.x == 0) {
    s_mean = sum_val + 1e-6f;
    s_mean = __fdividef(1.0f, s_mean);
  }
  __syncthreads();

  for (int i = 0; blockDim.x * i + threadIdx.x < seq_len; i++) {
    qk_offset =
        ((blockIdx.y + blockIdx.z)) * seq_len + blockDim.x * i + threadIdx.x;
    qk_buf_[qk_offset] = (T)(data[i] * s_mean);
  }
#endif
}

template <typename T, int ITEMS_PER_THREAD>
__global__ void softmax_kernel_v4_half2(T* qk_buf_,
                                        const T* attr_mask,
                                        const int batch_size,
                                        const int seq_len) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using T2 = half2;
  T2* qk_buf_half2 = reinterpret_cast<T2*>(qk_buf_);
  const T2* attr_mask_half2 = (const T2*)attr_mask;

  T2 data[ITEMS_PER_THREAD];
  int qk_offset;
  __shared__ float s_mean, s_max;
  float local_max = -1e20f;
  for (int i = 0;
       blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
       i++) {
    qk_offset = ((blockIdx.y + blockIdx.z)) * (seq_len / 2) + blockDim.x * i +
                threadIdx.x;
    int mask_offset = blockIdx.y * (seq_len / 2) + blockDim.x * i + threadIdx.x;

    T2 qk = qk_buf_half2[qk_offset];
    T2 mask_val = __ldg(&attr_mask_half2[mask_offset]);
    mask_val = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val),
                       __float2half2_rn(-10000.0f));

    data[i] = __hadd2(qk, mask_val);

    local_max = fmax(
        local_max,
        fmax(static_cast<float>(data[i].x), static_cast<float>(data[i].y)));
  }

  float max_val =
      blockDim.x <= 32
          ? phi::funcs::WarpReduceMax<float>(local_max, 0xFFFFFFFF)
          : phi::funcs::BlockReduceMax<float>(local_max, 0xFFFFFFFF);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();

  float local_sum = 0;
  for (int i = 0;
       blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
       i++) {
    data[i] = h2exp(__hsub2(data[i], __float2half2_rn(s_max)));
    local_sum += static_cast<float>(data[i].x + data[i].y);
  }

  float sum_val =
      blockDim.x <= 32
          ? phi::funcs::WarpReduceSum<float>(local_sum, 0xFFFFFFFF)
          : phi::funcs::BlockReduceSum<float>(local_sum, 0xFFFFFFFF);

  if (threadIdx.x == 0) {
    s_mean = sum_val + 1e-6f;
    s_mean = __fdividef(1.0f, s_mean);
  }
  __syncthreads();

  for (int i = 0;
       blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
       i++) {
    qk_offset = ((blockIdx.y + blockIdx.z)) * (seq_len / 2) + blockDim.x * i +
                threadIdx.x;
    qk_buf_half2[qk_offset] = __hmul2(data[i], __float2half2_rn(s_mean));
  }
#endif
}

template <typename T, int ITEMS_PER_THREAD, int NUM>
__global__ void softmax_kernel_v5_half2(T* qk_buf_,
                                        const T* attr_mask,
                                        const int batch_size,
                                        const int seq_len) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using T2 = half2;
  T2* qk_buf_half2 = reinterpret_cast<T2*>(qk_buf_);
  const T2* attr_mask_half2 = (const T2*)attr_mask;

  T2 data[NUM][ITEMS_PER_THREAD];

  int qk_offset[NUM];

  __shared__ float s_sum[NUM], s_max[NUM];
  float local_max[NUM];
#pragma unroll
  for (int j = 0; j < NUM; j++) {
    local_max[j] = -1e20f;
  }

  const int MAX_NUM = min((1 + gridDim.x - 1) / gridDim.x, NUM);
  for (int i = 0;
       blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
       i++) {
    int mask_offset[NUM];
#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      qk_offset[j] =
          ((blockIdx.y + blockIdx.z) + j * gridDim.x) * (seq_len / 2) +
          blockDim.x * i + threadIdx.x;
      mask_offset[j] = (blockIdx.y + j * gridDim.x) * (seq_len / 2) +
                       blockDim.x * i + threadIdx.x;
    }

    T2 mask_val[NUM];
#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      mask_val[j] = __ldg(&attr_mask_half2[mask_offset[j]]);
    }

    T2 qk[NUM];
#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      qk[j] = qk_buf_half2[qk_offset[j]];
    }
#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      mask_val[j] = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val[j]),
                            __float2half2_rn(-10000.0f));
    }
#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      data[j][i] = __hadd2(qk[j], mask_val[j]);
      local_max[j] = fmax(local_max[j],
                          fmax(static_cast<float>(data[j][i].x),
                               static_cast<float>(data[j][i].y)));
    }
  }
  if (blockDim.x <= 32) {
    phi::funcs::WarpReduceMaxV2<float, NUM>(local_max);
  } else {
    phi::funcs::BlockReduceMaxV2<float, NUM>(local_max);
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < NUM; j++) {
      s_max[j] = local_max[j];
    }
  }
  __syncthreads();
  float local_sum[NUM];
#pragma unroll
  for (int j = 0; j < NUM; j++) {
    local_sum[j] = {0.f};
  }

  for (int i = 0;
       blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
       i++) {
#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      data[j][i] = h2exp(__hsub2(data[j][i], __float2half2_rn(s_max[j])));
    }

#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      local_sum[j] += static_cast<float>(data[j][i].x + data[j][i].y);
    }
  }

  if (blockDim.x <= 32) {
    phi::funcs::WarpReduceSumV2<float, NUM>(local_sum);

  } else {
    phi::funcs::BlockReduceSumV2<float, NUM>(local_sum);
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < NUM; j++) {
      s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
    }
  }
  __syncthreads();

  for (int i = 0;
       blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD;
       i++) {
#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      qk_offset[j] =
          ((blockIdx.y + blockIdx.z) + j * gridDim.x) * (seq_len / 2) +
          blockDim.x * i + threadIdx.x;
    }

#pragma unroll
    for (int j = 0; j < MAX_NUM; j++) {
      qk_buf_half2[qk_offset[j]] =
          __hmul2(data[j][i], __float2half2_rn(s_sum[j]));
    }
  }
#endif
}

// --------      transpose_kernel      -------- //
template <typename T>
__global__ void transposeAxis01(
    T* out, T* in, const int dim0, const int dim1, const int dim2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dim0 * dim1 * dim2) {
    const int input_dim2_index = index % dim2;
    index = (index - input_dim2_index) / dim2;
    const int input_dim1_index = index % dim1;
    index = (index - input_dim1_index) / dim1;
    const int input_dim0_index = index % dim0;

    out[input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 +
        input_dim2_index] = in[input_dim0_index * dim1 * dim2 +
                               input_dim1_index * dim2 + input_dim2_index];
  }
}

// --------      padding_kernel      -------- //
template <typename T>
__global__ void paddingKernel(T* output1,
                              int* output2,
                              const T* input1,
                              const int* input2,
                              const int* input_lengths,
                              const int num_tokens,
                              const int batch_size,
                              const int max_seq_len,
                              const int num_experts) {
  const bool IS_FP16 = std::is_same<T, phi::dtype::float16>::value;
  const T MIN_T_VAL = (IS_FP16) ? (T)HALF_FLT_MIN : (T)FLT_MIN;
  int offset1 = blockIdx.x * num_tokens;
  int offset2 = blockIdx.x * batch_size * max_seq_len;
  for (int i = 0; i < batch_size; i++) {
    const T* in1_ptr = input1 + offset1;
    const int* in2_ptr = input2 + offset1;
    int input_length = input_lengths[i];
    offset1 += input_length;

    T* out1_ptr = output1 + offset2;
    int* out2_ptr = output2 + offset2;
    offset2 += max_seq_len;

    for (int j = threadIdx.x; j < max_seq_len; j += max_seq_len) {
      if (j < input_length) {
        out1_ptr[j] = in1_ptr[j];
        out2_ptr[j] = in2_ptr[j];
      } else {
        out1_ptr[j] = MIN_T_VAL;
        out2_ptr[j] = 0;
      }
    }
  }
}

// --------      general_topk_pair_sort_kernel      -------- //
template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void general_topk_pair_sort(T* out_keys,
                                       int* out_values,
                                       T* in_keys,
                                       int* in_values) {
  typedef cub::BlockRadixSort<T, BLOCK_THREADS, ITEMS_PER_THREAD, int>
      BlockRadixSort;
  typedef cub::
      BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>
          BlockLoadKey;
  typedef cub::
      BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>
          BlockLoadValue;
  typedef cub::
      BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE>
          BlockStoreKey;
  typedef cub::BlockStore<int,
                          BLOCK_THREADS,
                          ITEMS_PER_THREAD,
                          cub::BLOCK_STORE_TRANSPOSE>
      BlockStoreValue;

  __shared__ union {
    typename BlockRadixSort::TempStorage sort;
    typename BlockLoadKey::TempStorage loadkey;
    typename BlockLoadValue::TempStorage loadvalue;
    typename BlockStoreKey::TempStorage storekey;
    typename BlockStoreValue::TempStorage storevalue;
  } temp_storage;

  int block_offset = blockIdx.x * BLOCK_THREADS * ITEMS_PER_THREAD;

  T thread_keys[ITEMS_PER_THREAD];
  int thread_values[ITEMS_PER_THREAD];
  BlockLoadKey(temp_storage.loadkey).Load(in_keys + block_offset, thread_keys);
  BlockLoadValue(temp_storage.loadvalue)
      .Load(in_values + block_offset, thread_values);
  __syncthreads();

  BlockRadixSort(temp_storage.sort).SortDescending(thread_keys, thread_values);
  __syncthreads();

  BlockStoreKey(temp_storage.storekey)
      .Store(out_keys + block_offset, thread_keys);
  BlockStoreValue(temp_storage.storevalue)
      .Store(out_values + block_offset, thread_values);
}

// --------      finalize_moe_routing_kernel      -------- //
template <typename T>
__global__ void finalize_moe_routing_kernel(
    const T* expanded_permuted_rows,
    T* reduced_unpermuted_output,
    const T* skip,
    const T* bias,
    const T* scales,
    const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row,
    const int cols,
    const int k,
    bool ec_route) {
  const int original_row = blockIdx.x;
  const int num_rows = gridDim.x;
  T* reduced_row_ptr = reduced_unpermuted_output + original_row * cols;
  const T* skip_row_ptr = skip + original_row * cols;

  for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
    T thread_output = skip_row_ptr[tid];
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int expanded_original_row = original_row + k_idx * num_rows;
      const int expanded_permuted_row =
          expanded_source_row_to_expanded_dest_row[expanded_original_row];

      if (ec_route && expanded_permuted_row == -1) continue;
      const int64_t k_offset =
          ec_route ? expanded_original_row : original_row * k + k_idx;
      const T row_scale = scales[k_offset];
      const T* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows + expanded_permuted_row * cols;

      const int expert_idx = ec_route ? k_idx : expert_for_source_row[k_offset];
      const T* bias_ptr = bias + expert_idx * cols;

      thread_output =
          thread_output +
          row_scale * (expanded_permuted_rows_row_ptr[tid] + bias_ptr[tid]);
    }
    reduced_row_ptr[tid] = thread_output;
  }
}

// --------      initialize_moe_routing_kernel      -------- //
template <typename T, int VecSize>
__global__ void initialize_moe_routing_kernel(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int num_rows,
    const int active_rows,
    const int cols,
    const int k,
    const int max_seq_len,
    bool ec_route) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  // Reverse permutation map.
  // I do this so that later, we can use the source -> dest map to do the k-way
  // reduction and unpermuting. I need the reverse map for that reduction to
  // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
  // thread block will be responsible for all k summations.
  const int expanded_dest_row = blockIdx.x;
  const int expanded_source_row =
      ec_route ? expanded_dest_row_to_expanded_source_row[expanded_dest_row /
                                                              k * max_seq_len +
                                                          expanded_dest_row % k]
               : expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  if (threadIdx.x == 0) {
    expanded_source_row_to_expanded_dest_row[expanded_source_row] =
        expanded_dest_row;
  }

  if (blockIdx.x < active_rows) {
    // Duplicate and permute rows
    const int source_row = expanded_source_row % num_rows;

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    T* dest_row_ptr = permuted_output + expanded_dest_row * cols;

    for (int tid = threadIdx.x * VecSize; tid < cols;
         tid += blockDim.x * VecSize) {
      // dest_row_ptr[tid] = source_row_ptr[tid];
      phi::Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
      phi::Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
    }
  }
}

}  // namespace phi
