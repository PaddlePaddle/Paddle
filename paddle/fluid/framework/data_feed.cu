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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)

#include "paddle/fluid/framework/data_feed.h"
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <sstream>
#include "cub/cub.cuh"
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_GPU_GRAPH)
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#endif
#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/gpu/graph_reindex_funcs.h"
#include "paddle/phi/kernels/graph_reindex_kernel.h"

PHI_DECLARE_bool(enable_opt_get_features);
PHI_DECLARE_bool(graph_metapath_split_opt);
PHI_DECLARE_int32(gpugraph_storage_mode);
PHI_DECLARE_double(gpugraph_hbm_table_load_factor);

namespace paddle {
namespace framework {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define DEBUG_STATE(state)                                             \
  VLOG(2) << "left: " << state->left << " right: " << state->right     \
          << " central_word: " << state->central_word                  \
          << " step: " << state->step << " cursor: " << state->cursor  \
          << " len: " << state->len << " row_num: " << state->row_num; \
// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void fill_idx(T *idx, size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    idx[i] = i;
  }
}

/**
 * @brief sort cub
 */
template <typename K, typename V>
void cub_sort_pairs(int len,
                    const K *in_keys,
                    K *out_keys,
                    const V *in_vals,
                    V *out_vals,
                    cudaStream_t stream,
                    std::shared_ptr<phi::Allocation> &d_buf_,  // NOLINT
                    const paddle::platform::Place &place_) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL,
                                             temp_storage_bytes,
                                             in_keys,
                                             out_keys,
                                             in_vals,
                                             out_vals,
                                             len,
                                             0,
                                             8 * sizeof(K),
                                             stream,
                                             false));
  if (d_buf_ == NULL || d_buf_->size() < temp_storage_bytes) {
    d_buf_ = memory::AllocShared(
        place_,
        temp_storage_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  }
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_buf_->ptr(),
                                             temp_storage_bytes,
                                             in_keys,
                                             out_keys,
                                             in_vals,
                                             out_vals,
                                             len,
                                             0,
                                             8 * sizeof(K),
                                             stream,
                                             false));
}

/**
 * @Brief cub run length encode
 */
template <typename K, typename V, typename TNum>
void cub_runlength_encode(int N,
                          const K *in_keys,
                          K *out_keys,
                          V *out_sizes,
                          TNum *d_out_len,
                          cudaStream_t stream,
                          std::shared_ptr<phi::Allocation> &d_buf_,  // NOLINT
                          const paddle::platform::Place &place_) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(NULL,
                                                temp_storage_bytes,
                                                in_keys,
                                                out_keys,
                                                out_sizes,
                                                d_out_len,
                                                N,
                                                stream));
  if (d_buf_ == NULL || d_buf_->size() < temp_storage_bytes) {
    d_buf_ = memory::AllocShared(
        place_,
        temp_storage_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  }
  CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_buf_->ptr(),
                                                temp_storage_bytes,
                                                in_keys,
                                                out_keys,
                                                out_sizes,
                                                d_out_len,
                                                N,
                                                stream));
}

/**
 * @brief exclusive sum
 */
template <typename K>
void cub_exclusivesum(int N,
                      const K *in,
                      K *out,
                      cudaStream_t stream,
                      std::shared_ptr<phi::Allocation> &d_buf_,  // NOLINT
                      const paddle::platform::Place &place_) {
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      NULL, temp_storage_bytes, in, out, N, stream));
  if (d_buf_ == NULL || d_buf_->size() < temp_storage_bytes) {
    d_buf_ = memory::AllocShared(
        place_,
        temp_storage_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  }
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      d_buf_->ptr(), temp_storage_bytes, in, out, N, stream));
}

template <typename T>
__global__ void kernel_fill_restore_idx(size_t N,
                                        const T *d_sorted_idx,
                                        const T *d_offset,
                                        const T *d_merged_cnts,
                                        T *d_restore_idx) {
  CUDA_KERNEL_LOOP(i, N) {
    const T &off = d_offset[i];
    const T &num = d_merged_cnts[i];
    for (size_t k = 0; k < num; k++) {
      d_restore_idx[d_sorted_idx[off + k]] = i;
    }
  }
}

template <typename T>
__global__ void kernel_fill_restore_idx_by_search(size_t N,
                                                  const T *d_sorted_idx,
                                                  size_t merge_num,
                                                  const T *d_offset,
                                                  T *d_restore_idx) {
  CUDA_KERNEL_LOOP(i, N) {
    if (i < d_offset[1]) {
      d_restore_idx[d_sorted_idx[i]] = 0;
      continue;
    }
    int high = merge_num - 1;
    int low = 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < d_offset[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    d_restore_idx[d_sorted_idx[i]] = low;
  }
}

// For unique node and inverse id.
int dedup_keys_and_fillidx(int total_nodes_num,
                           const uint64_t *d_keys,
                           uint64_t *d_merged_keys,  // input
                           uint64_t *d_sorted_keys,  // output
                           uint32_t *d_restore_idx,  // inverse
                           uint32_t *d_sorted_idx,
                           uint32_t *d_offset,
                           uint32_t *d_merged_cnts,
                           cudaStream_t stream,
                           std::shared_ptr<phi::Allocation> &d_buf_,  // NOLINT
                           const paddle::platform::Place &place_) {
  int merged_size = 0;  // Final num
  auto d_index_in =
      memory::Alloc(place_,
                    sizeof(uint32_t) * (total_nodes_num + 1),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t *d_index_in_ptr = reinterpret_cast<uint32_t *>(d_index_in->ptr());
  int *d_merged_size =
      reinterpret_cast<int *>(&d_index_in_ptr[total_nodes_num]);
  fill_idx<<<GET_BLOCKS(total_nodes_num), CUDA_NUM_THREADS, 0, stream>>>(
      d_index_in_ptr, total_nodes_num);
  cub_sort_pairs(total_nodes_num,
                 d_keys,
                 d_sorted_keys,
                 d_index_in_ptr,
                 d_sorted_idx,
                 stream,
                 d_buf_,
                 place_);
  cub_runlength_encode(total_nodes_num,
                       d_sorted_keys,
                       d_merged_keys,
                       d_merged_cnts,
                       d_merged_size,
                       stream,
                       d_buf_,
                       place_);
  CUDA_CHECK(cudaMemcpyAsync(&merged_size,
                             d_merged_size,
                             sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cub_exclusivesum(
      merged_size, d_merged_cnts, d_offset, stream, d_buf_, place_);

  if (total_nodes_num < merged_size * 2) {
    kernel_fill_restore_idx<<<GET_BLOCKS(merged_size),
                              CUDA_NUM_THREADS,
                              0,
                              stream>>>(
        merged_size, d_sorted_idx, d_offset, d_merged_cnts, d_restore_idx);
  } else {
    // used mid search fill idx when high dedup rate
    kernel_fill_restore_idx_by_search<<<GET_BLOCKS(total_nodes_num),
                                        CUDA_NUM_THREADS,
                                        0,
                                        stream>>>(
        total_nodes_num, d_sorted_idx, merged_size, d_offset, d_restore_idx);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return merged_size;
}

// fill slot values
__global__ void FillSlotValueOffsetKernel(const int ins_num,
                                          const int used_slot_num,
                                          size_t *slot_value_offsets,
                                          const int *uint64_offsets,
                                          const int uint64_slot_size,
                                          const int *float_offsets,
                                          const int float_slot_size,
                                          const UsedSlotGpuType *used_slots) {
  int col_num = ins_num + 1;
  int uint64_cols = uint64_slot_size + 1;
  int float_cols = float_slot_size + 1;

  CUDA_KERNEL_LOOP(slot_idx, used_slot_num) {
    int value_off = slot_idx * col_num;
    slot_value_offsets[value_off] = 0;

    auto &info = used_slots[slot_idx];
    if (info.is_uint64_value) {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * uint64_cols + info.slot_value_idx;
        int num = uint64_offsets[pos + 1] - uint64_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    } else {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * float_cols + info.slot_value_idx;
        int num = float_offsets[pos + 1] - float_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    }
  }
}

void SlotRecordInMemoryDataFeed::FillSlotValueOffset(
    const int ins_num,
    const int used_slot_num,
    size_t *slot_value_offsets,
    const int *uint64_offsets,
    const int uint64_slot_size,
    const int *float_offsets,
    const int float_slot_size,
    const UsedSlotGpuType *used_slots,
    cudaStream_t stream) {
  FillSlotValueOffsetKernel<<<GET_BLOCKS(used_slot_num),
                              CUDA_NUM_THREADS,
                              0,
                              stream>>>(ins_num,
                                        used_slot_num,
                                        slot_value_offsets,
                                        uint64_offsets,
                                        uint64_slot_size,
                                        float_offsets,
                                        float_slot_size,
                                        used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void CopyForTensorKernel(const int used_slot_num,
                                    const int ins_num,
                                    void **dest,
                                    const size_t *slot_value_offsets,
                                    const uint64_t *uint64_feas,
                                    const int *uint64_offsets,
                                    const int *uint64_ins_lens,
                                    const int uint64_slot_size,
                                    const float *float_feas,
                                    const int *float_offsets,
                                    const int *float_ins_lens,
                                    const int float_slot_size,
                                    const UsedSlotGpuType *used_slots) {
  int col_num = ins_num + 1;
  int uint64_cols = uint64_slot_size + 1;
  int float_cols = float_slot_size + 1;

  CUDA_KERNEL_LOOP(i, ins_num * used_slot_num) {
    int slot_idx = i / ins_num;
    int ins_idx = i % ins_num;

    uint32_t value_offset = slot_value_offsets[slot_idx * col_num + ins_idx];
    auto &info = used_slots[slot_idx];
    if (info.is_uint64_value) {
      uint64_t *up = reinterpret_cast<uint64_t *>(dest[slot_idx]);
      int index = info.slot_value_idx + uint64_cols * ins_idx;
      int old_off = uint64_offsets[index];
      int num = uint64_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
      int uint64_value_offset = uint64_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        up[k + value_offset] = uint64_feas[k + old_off + uint64_value_offset];
      }
    } else {
      float *fp = reinterpret_cast<float *>(dest[slot_idx]);
      int index = info.slot_value_idx + float_cols * ins_idx;
      int old_off = float_offsets[index];
      int num = float_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
      int float_value_offset = float_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        fp[k + value_offset] = float_feas[k + old_off + float_value_offset];
      }
    }
  }
}

void SlotRecordInMemoryDataFeed::CopyForTensor(
    const int ins_num,
    const int used_slot_num,
    void **dest,
    const size_t *slot_value_offsets,
    const uint64_t *uint64_feas,
    const int *uint64_offsets,
    const int *uint64_ins_lens,
    const int uint64_slot_size,
    const float *float_feas,
    const int *float_offsets,
    const int *float_ins_lens,
    const int float_slot_size,
    const UsedSlotGpuType *used_slots,
    cudaStream_t stream) {
  CopyForTensorKernel<<<GET_BLOCKS(used_slot_num * ins_num),
                        CUDA_NUM_THREADS,
                        0,
                        stream>>>(used_slot_num,
                                  ins_num,
                                  dest,
                                  slot_value_offsets,
                                  uint64_feas,
                                  uint64_offsets,
                                  uint64_ins_lens,
                                  uint64_slot_size,
                                  float_feas,
                                  float_offsets,
                                  float_ins_lens,
                                  float_slot_size,
                                  used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void GraphFillCVMKernel(int64_t *tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { tensor[idx] = 1; }
}

__global__ void CopyDuplicateKeys(int64_t *dist_tensor,
                                  uint64_t *src_tensor,
                                  int len) {
  CUDA_KERNEL_LOOP(idx, len) {
    dist_tensor[idx * 2] = src_tensor[idx];
    dist_tensor[idx * 2 + 1] = src_tensor[idx];
  }
}

#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_GPU_GRAPH)
int GraphDataGenerator::AcquireInstance(BufState *state) {
  if (state->GetNextStep()) {
    DEBUG_STATE(state);
    return state->len;
  } else if (state->GetNextCentrolWord()) {
    DEBUG_STATE(state);
    return state->len;
  } else if (state->GetNextBatch()) {
    DEBUG_STATE(state);
    return state->len;
  }
  return 0;
}

__global__ void GraphFillIdKernel(uint64_t *id_tensor,
                                  int *fill_ins_num,
                                  uint64_t *walk,
                                  uint8_t *walk_ntype,
                                  int *row,
                                  int central_word,
                                  int step,
                                  int len,
                                  int col_num,
                                  uint8_t *excluded_train_pair,
                                  int excluded_train_pair_len) {
  __shared__ uint64_t local_key[CUDA_NUM_THREADS * 2];
  __shared__ int local_num;
  __shared__ int global_num;
  bool need_filter = false;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();
  // int dst = idx * 2;
  // id_tensor[dst] = walk[src];
  // id_tensor[dst + 1] = walk[src + step];
  if (idx < len) {
    int src = row[idx] * col_num + central_word;
    if (walk[src] != 0 && walk[src + step] != 0) {
      for (int i = 0; i < excluded_train_pair_len; i += 2) {
        if (walk_ntype[src] == excluded_train_pair[i] &&
            walk_ntype[src + step] == excluded_train_pair[i + 1]) {
          // filter this pair
          need_filter = true;
          break;
        }
      }
      if (!need_filter) {
        size_t dst = atomicAdd(&local_num, 1);
        local_key[dst * 2] = walk[src];
        local_key[dst * 2 + 1] = walk[src + step];
      }
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(fill_ins_num, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    id_tensor[global_num * 2 + 2 * threadIdx.x] = local_key[2 * threadIdx.x];
    id_tensor[global_num * 2 + 2 * threadIdx.x + 1] =
        local_key[2 * threadIdx.x + 1];
  }
}

__global__ void GraphFillSlotKernel(uint64_t *id_tensor,
                                    uint64_t *feature_buf,
                                    int len,
                                    int total_ins,
                                    int slot_num,
                                    int *slot_feature_num_map,
                                    int fea_num_per_node,
                                    int *actual_slot_id_map,
                                    int *fea_offset_map) {
  CUDA_KERNEL_LOOP(idx, len) {
    int fea_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    int actual_slot_id = actual_slot_id_map[fea_idx];
    int fea_offset = fea_offset_map[fea_idx];
    reinterpret_cast<uint64_t *>(id_tensor[actual_slot_id])
        [ins_idx * slot_feature_num_map[actual_slot_id] + fea_offset] =
            feature_buf[ins_idx * fea_num_per_node + fea_idx];
  }
}

__global__ void GraphFillSlotLodKernelOpt(uint64_t *id_tensor,
                                          int len,
                                          int total_ins,
                                          int *slot_feature_num_map) {
  CUDA_KERNEL_LOOP(idx, len) {
    int slot_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    (reinterpret_cast<uint64_t *>(id_tensor[slot_idx]))[ins_idx] =
        ins_idx * slot_feature_num_map[slot_idx];
  }
}

__global__ void GraphFillSlotLodKernel(int64_t *id_tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { id_tensor[idx] = idx; }
}

// fill sage neighbor results
__global__ void FillActualNeighbors(int64_t *vals,
                                    int64_t *actual_vals,
                                    int64_t *actual_vals_dst,
                                    int *actual_sample_size,
                                    int *cumsum_actual_sample_size,
                                    int sample_size,
                                    int len,
                                    int mod) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    int offset1 = cumsum_actual_sample_size[i];
    int offset2 = sample_size * i;
    int dst_id = i % mod;
    for (int j = 0; j < actual_sample_size[i]; j++) {
      actual_vals[offset1 + j] = vals[offset2 + j];
      actual_vals_dst[offset1 + j] = dst_id;
    }
  }
}

int GraphDataGenerator::FillIdShowClkTensor(int total_instance,
                                            bool gpu_graph_training,
                                            size_t cursor) {
  id_tensor_ptr_ =
      feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
  show_tensor_ptr_ =
      feed_vec_[1]->mutable_data<int64_t>({total_instance}, this->place_);
  clk_tensor_ptr_ =
      feed_vec_[2]->mutable_data<int64_t>({total_instance}, this->place_);
  if (gpu_graph_training) {
    uint64_t *ins_cursor, *ins_buf;
    ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
    ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
    cudaMemcpyAsync(id_tensor_ptr_,
                    ins_cursor,
                    sizeof(uint64_t) * total_instance,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);
  } else {
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys_[cursor]->ptr());
    d_type_keys += infer_node_start_;
    infer_node_start_ += total_instance / 2;
    CopyDuplicateKeys<<<GET_BLOCKS(total_instance / 2),
                        CUDA_NUM_THREADS,
                        0,
                        train_stream_>>>(
        id_tensor_ptr_, d_type_keys, total_instance / 2);
  }

  GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(show_tensor_ptr_, total_instance);
  GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(clk_tensor_ptr_, total_instance);
  return 0;
}

int GraphDataGenerator::FillGraphIdShowClkTensor(int uniq_instance,
                                                 int total_instance,
                                                 int index) {
  id_tensor_ptr_ =
      feed_vec_[0]->mutable_data<int64_t>({uniq_instance, 1}, this->place_);
  show_tensor_ptr_ =
      feed_vec_[1]->mutable_data<int64_t>({uniq_instance}, this->place_);
  clk_tensor_ptr_ =
      feed_vec_[2]->mutable_data<int64_t>({uniq_instance}, this->place_);
  int index_offset = 3 + slot_num_ * 2 + 5 * samples_.size();
  index_tensor_ptr_ = feed_vec_[index_offset]->mutable_data<int>(
      {total_instance}, this->place_);
  if (get_degree_) {
    degree_tensor_ptr_ = feed_vec_[index_offset + 1]->mutable_data<int>(
        {uniq_instance * edge_to_id_len_}, this->place_);
  }

  int len_samples = samples_.size();
  int *num_nodes_tensor_ptr_[len_samples];
  int *next_num_nodes_tensor_ptr_[len_samples];
  int64_t *edges_src_tensor_ptr_[len_samples];
  int64_t *edges_dst_tensor_ptr_[len_samples];
  int *edges_split_tensor_ptr_[len_samples];

  std::vector<std::vector<int>> edges_split_num_for_graph =
      edges_split_num_vec_[index];
  std::vector<std::shared_ptr<phi::Allocation>> graph_edges =
      graph_edges_vec_[index];
  for (int i = 0; i < len_samples; i++) {
    int offset = 3 + 2 * slot_num_ + 5 * i;
    std::vector<int> edges_split_num = edges_split_num_for_graph[i];

    int neighbor_len = edges_split_num[edge_to_id_len_ + 2];
    num_nodes_tensor_ptr_[i] =
        feed_vec_[offset]->mutable_data<int>({1}, this->place_);
    next_num_nodes_tensor_ptr_[i] =
        feed_vec_[offset + 1]->mutable_data<int>({1}, this->place_);
    edges_src_tensor_ptr_[i] = feed_vec_[offset + 2]->mutable_data<int64_t>(
        {neighbor_len, 1}, this->place_);
    edges_dst_tensor_ptr_[i] = feed_vec_[offset + 3]->mutable_data<int64_t>(
        {neighbor_len, 1}, this->place_);
    edges_split_tensor_ptr_[i] = feed_vec_[offset + 4]->mutable_data<int>(
        {edge_to_id_len_}, this->place_);

    // [edges_split_num, next_num_nodes, num_nodes, neighbor_len]
    cudaMemcpyAsync(next_num_nodes_tensor_ptr_[i],
                    edges_split_num.data() + edge_to_id_len_,
                    sizeof(int),
                    cudaMemcpyHostToDevice,
                    train_stream_);
    cudaMemcpyAsync(num_nodes_tensor_ptr_[i],
                    edges_split_num.data() + edge_to_id_len_ + 1,
                    sizeof(int),
                    cudaMemcpyHostToDevice,
                    train_stream_);
    cudaMemcpyAsync(edges_split_tensor_ptr_[i],
                    edges_split_num.data(),
                    sizeof(int) * edge_to_id_len_,
                    cudaMemcpyHostToDevice,
                    train_stream_);
    cudaMemcpyAsync(edges_src_tensor_ptr_[i],
                    graph_edges[i * 2]->ptr(),
                    sizeof(int64_t) * neighbor_len,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);
    cudaMemcpyAsync(edges_dst_tensor_ptr_[i],
                    graph_edges[i * 2 + 1]->ptr(),
                    sizeof(int64_t) * neighbor_len,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);
  }

  cudaMemcpyAsync(id_tensor_ptr_,
                  final_sage_nodes_vec_[index]->ptr(),
                  sizeof(int64_t) * uniq_instance,
                  cudaMemcpyDeviceToDevice,
                  train_stream_);
  cudaMemcpyAsync(index_tensor_ptr_,
                  inverse_vec_[index]->ptr(),
                  sizeof(int) * total_instance,
                  cudaMemcpyDeviceToDevice,
                  train_stream_);
  if (get_degree_) {
    cudaMemcpyAsync(degree_tensor_ptr_,
                    node_degree_vec_[index]->ptr(),
                    sizeof(int) * uniq_instance * edge_to_id_len_,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);
  }
  GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(show_tensor_ptr_, uniq_instance);
  GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(clk_tensor_ptr_, uniq_instance);
  return 0;
}

int GraphDataGenerator::FillGraphSlotFeature(
    int total_instance,
    bool gpu_graph_training,
    std::shared_ptr<phi::Allocation> final_sage_nodes) {
  uint64_t *ins_cursor, *ins_buf;
  if (gpu_graph_training) {
    ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
    ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
  } else {
    id_tensor_ptr_ =
        feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
    ins_cursor = reinterpret_cast<uint64_t *>(id_tensor_ptr_);
  }

  if (!sage_mode_) {
    return FillSlotFeature(ins_cursor, total_instance);
  } else {
    uint64_t *sage_nodes_ptr =
        reinterpret_cast<uint64_t *>(final_sage_nodes->ptr());
    return FillSlotFeature(sage_nodes_ptr, total_instance);
  }
}

int GraphDataGenerator::MakeInsPair(cudaStream_t stream) {
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_->ptr());
  uint8_t *walk_ntype = NULL;
  uint8_t *excluded_train_pair = NULL;
  if (excluded_train_pair_len_ > 0) {
    walk_ntype = reinterpret_cast<uint8_t *>(d_walk_ntype_->ptr());
    excluded_train_pair =
        reinterpret_cast<uint8_t *>(d_excluded_train_pair_->ptr());
  }
  uint64_t *ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
  int *random_row = reinterpret_cast<int *>(d_random_row_->ptr());
  int *d_pair_num = reinterpret_cast<int *>(d_pair_num_->ptr());
  cudaMemsetAsync(d_pair_num, 0, sizeof(int), stream);
  int len = buf_state_.len;
  // make pair
  GraphFillIdKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream>>>(
      ins_buf + ins_buf_pair_len_ * 2,
      d_pair_num,
      walk,
      walk_ntype,
      random_row + buf_state_.cursor,
      buf_state_.central_word,
      window_step_[buf_state_.step],
      len,
      walk_len_,
      excluded_train_pair,
      excluded_train_pair_len_);
  int h_pair_num;
  cudaMemcpyAsync(
      &h_pair_num, d_pair_num, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  ins_buf_pair_len_ += h_pair_num;

  if (debug_mode_) {
    uint64_t h_ins_buf[ins_buf_pair_len_ * 2];  // NOLINT
    cudaMemcpy(h_ins_buf,
               ins_buf,
               2 * ins_buf_pair_len_ * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    VLOG(2) << "h_pair_num = " << h_pair_num
            << ", ins_buf_pair_len = " << ins_buf_pair_len_;
    for (int xx = 0; xx < ins_buf_pair_len_; xx++) {
      VLOG(2) << "h_ins_buf: " << h_ins_buf[xx * 2] << ", "
              << h_ins_buf[xx * 2 + 1];
    }
  }
  return ins_buf_pair_len_;
}

int GraphDataGenerator::FillInsBuf(cudaStream_t stream) {
  if (ins_buf_pair_len_ >= batch_size_) {
    return batch_size_;
  }
  int total_instance = AcquireInstance(&buf_state_);

  VLOG(2) << "total_ins: " << total_instance;
  buf_state_.Debug();

  if (total_instance == 0) {
    return -1;
  }
  return MakeInsPair(stream);
}

int GraphDataGenerator::GenerateBatch() {
  int total_instance = 0;
  platform::CUDADeviceGuard guard(gpuid_);
  int res = 0;
  if (!gpu_graph_training_) {
    // infer
    if (!sage_mode_) {
      total_instance = (infer_node_start_ + batch_size_ <= infer_node_end_)
                           ? batch_size_
                           : infer_node_end_ - infer_node_start_;
      VLOG(1) << "in graph_data generator:batch_size = " << batch_size_
              << " instance = " << total_instance;
      total_instance *= 2;
      if (total_instance == 0) {
        return 0;
      }
      FillIdShowClkTensor(total_instance, gpu_graph_training_, cursor_);
    } else {
      if (sage_batch_count_ == sage_batch_num_) {
        return 0;
      }
      FillGraphIdShowClkTensor(uniq_instance_vec_[sage_batch_count_],
                               total_instance_vec_[sage_batch_count_],
                               sage_batch_count_);
    }
  } else {
    // train
    if (!sage_mode_) {
      while (ins_buf_pair_len_ < batch_size_) {
        res = FillInsBuf(train_stream_);
        if (res == -1) {
          if (ins_buf_pair_len_ == 0) {
            return 0;
          } else {
            break;
          }
        }
      }
      total_instance =
          ins_buf_pair_len_ < batch_size_ ? ins_buf_pair_len_ : batch_size_;
      total_instance *= 2;
      VLOG(2) << "total_instance: " << total_instance
              << ", ins_buf_pair_len = " << ins_buf_pair_len_;
      FillIdShowClkTensor(total_instance, gpu_graph_training_);
    } else {
      if (sage_batch_count_ == sage_batch_num_) {
        return 0;
      }
      FillGraphIdShowClkTensor(uniq_instance_vec_[sage_batch_count_],
                               total_instance_vec_[sage_batch_count_],
                               sage_batch_count_);
    }
  }

  if (slot_num_ > 0) {
    if (!sage_mode_) {
      FillGraphSlotFeature(total_instance, gpu_graph_training_);
    } else {
      FillGraphSlotFeature(uniq_instance_vec_[sage_batch_count_],
                           gpu_graph_training_,
                           final_sage_nodes_vec_[sage_batch_count_]);
    }
  }
  offset_.clear();
  offset_.push_back(0);
  if (!sage_mode_) {
    offset_.push_back(total_instance);
  } else {
    offset_.push_back(uniq_instance_vec_[sage_batch_count_]);
    sage_batch_count_ += 1;
  }
  LoD lod{offset_};
  feed_vec_[0]->set_lod(lod);
  if (slot_num_ > 0) {
    for (int i = 0; i < slot_num_; ++i) {
      feed_vec_[3 + 2 * i]->set_lod(lod);
    }
  }

  cudaStreamSynchronize(train_stream_);
  if (!gpu_graph_training_) return 1;
  if (!sage_mode_) {
    ins_buf_pair_len_ -= total_instance / 2;
  }
  return 1;
}

__global__ void GraphFillSampleKeysKernel(uint64_t *neighbors,
                                          uint64_t *sample_keys,
                                          int *prefix_sum,
                                          int *sampleidx2row,
                                          int *tmp_sampleidx2row,
                                          int *actual_sample_size,
                                          int cur_degree,
                                          int len) {
  CUDA_KERNEL_LOOP(idx, len) {
    for (int k = 0; k < actual_sample_size[idx]; k++) {
      size_t offset = prefix_sum[idx] + k;
      sample_keys[offset] = neighbors[idx * cur_degree + k];
      tmp_sampleidx2row[offset] = sampleidx2row[idx] + k;
    }
  }
}

__global__ void GraphDoWalkKernel(uint64_t *neighbors,
                                  uint64_t *walk,
                                  uint8_t *walk_ntype,
                                  int *d_prefix_sum,
                                  int *actual_sample_size,
                                  int cur_degree,
                                  int step,
                                  int len,
                                  int *id_cnt,
                                  int *sampleidx2row,
                                  int col_size,
                                  uint8_t edge_dst_id) {
  CUDA_KERNEL_LOOP(i, len) {
    for (int k = 0; k < actual_sample_size[i]; k++) {
      // int idx = sampleidx2row[i];
      size_t row = sampleidx2row[k + d_prefix_sum[i]];
      // size_t row = idx * cur_degree + k;
      size_t col = step;
      size_t offset = (row * col_size + col);
      walk[offset] = neighbors[i * cur_degree + k];
      if (walk_ntype != NULL) {
        walk_ntype[offset] = edge_dst_id;
      }
    }
  }
}

// Fill keys to the first column of walk
__global__ void GraphFillFirstStepKernel(int *prefix_sum,
                                         int *sampleidx2row,
                                         uint64_t *walk,
                                         uint8_t *walk_ntype,
                                         uint64_t *keys,
                                         uint8_t edge_src_id,
                                         uint8_t edge_dst_id,
                                         int len,
                                         int walk_degree,
                                         int col_size,
                                         int *actual_sample_size,
                                         uint64_t *neighbors,
                                         uint64_t *sample_keys) {
  CUDA_KERNEL_LOOP(idx, len) {
    for (int k = 0; k < actual_sample_size[idx]; k++) {
      size_t row = prefix_sum[idx] + k;
      sample_keys[row] = neighbors[idx * walk_degree + k];
      sampleidx2row[row] = row;

      size_t offset = col_size * row;
      walk[offset] = keys[idx];
      walk[offset + 1] = neighbors[idx * walk_degree + k];
      if (walk_ntype != NULL) {
        walk_ntype[offset] = edge_src_id;
        walk_ntype[offset + 1] = edge_dst_id;
      }
    }
  }
}

__global__ void get_each_ins_info(uint8_t *slot_list,
                                  uint32_t *slot_size_list,
                                  uint32_t *slot_size_prefix,
                                  uint32_t *each_ins_slot_num,
                                  uint32_t *each_ins_slot_num_inner_prefix,
                                  size_t key_num,
                                  int slot_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < key_num) {
    uint32_t slot_index = slot_size_prefix[i];
    size_t each_ins_slot_index = i * slot_num;
    for (int j = 0; j < slot_size_list[i]; j++) {
      each_ins_slot_num[each_ins_slot_index + slot_list[slot_index + j]] += 1;
    }
    each_ins_slot_num_inner_prefix[each_ins_slot_index] = 0;
    for (int j = 1; j < slot_num; j++) {
      each_ins_slot_num_inner_prefix[each_ins_slot_index + j] =
          each_ins_slot_num[each_ins_slot_index + j - 1] +
          each_ins_slot_num_inner_prefix[each_ins_slot_index + j - 1];
    }
  }
}

__global__ void fill_slot_num(uint32_t *d_each_ins_slot_num_ptr,
                              uint64_t **d_ins_slot_num_vector_ptr,
                              size_t key_num,
                              int slot_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < key_num) {
    size_t d_each_index = i * slot_num;
    for (int j = 0; j < slot_num; j++) {
      d_ins_slot_num_vector_ptr[j][i] =
          d_each_ins_slot_num_ptr[d_each_index + j];
    }
  }
}

__global__ void fill_slot_tensor(uint64_t *feature_list,
                                 uint32_t *feature_size_prefixsum,
                                 uint32_t *each_ins_slot_num_inner_prefix,
                                 uint64_t *ins_slot_num,
                                 int64_t *slot_lod_tensor,
                                 int64_t *slot_tensor,
                                 int slot,
                                 int slot_num,
                                 size_t node_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < node_num) {
    size_t dst_index = slot_lod_tensor[i];
    size_t src_index = feature_size_prefixsum[i] +
                       each_ins_slot_num_inner_prefix[slot_num * i + slot];
    for (uint64_t j = 0; j < ins_slot_num[i]; j++) {
      slot_tensor[dst_index + j] = feature_list[src_index + j];
    }
  }
}

__global__ void GetUniqueFeaNum(uint64_t *d_in,
                                uint64_t *unique_num,
                                size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint64_t local_num;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();

  if (i < len - 1) {
    if (d_in[i] != d_in[i + 1]) {
      atomicAdd(&local_num, 1);
    }
  }
  if (i == len - 1) {
    atomicAdd(&local_num, 1);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(unique_num, local_num);
  }
}

__global__ void UniqueFeature(uint64_t *d_in,
                              uint64_t *d_out,
                              uint64_t *unique_num,
                              size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint64_t local_key[CUDA_NUM_THREADS];
  __shared__ uint64_t local_num;
  __shared__ uint64_t global_num;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();

  if (i < len - 1) {
    if (d_in[i] != d_in[i + 1]) {
      size_t dst = atomicAdd(&local_num, 1);
      local_key[dst] = d_in[i];
    }
  }
  if (i == len - 1) {
    size_t dst = atomicAdd(&local_num, 1);
    local_key[dst] = d_in[i];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(unique_num, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    d_out[global_num + threadIdx.x] = local_key[threadIdx.x];
  }
}
// Fill sample_res to the stepth column of walk
void GraphDataGenerator::FillOneStep(uint64_t *d_start_ids,
                                     int etype_id,
                                     uint64_t *walk,
                                     uint8_t *walk_ntype,
                                     int len,
                                     NeighborSampleResult &sample_res,
                                     int cur_degree,
                                     int step,
                                     int *len_per_row) {
  platform::CUDADeviceGuard guard(gpuid_);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  uint64_t node_id = gpu_graph_ptr->edge_to_node_map_[etype_id];
  uint8_t edge_src_id = node_id >> 32;
  uint8_t edge_dst_id = node_id;
  size_t temp_storage_bytes = 0;
  int *d_actual_sample_size = sample_res.actual_sample_size;
  uint64_t *d_neighbors = sample_res.val;
  int *d_prefix_sum = reinterpret_cast<int *>(d_prefix_sum_->ptr());
  uint64_t *d_sample_keys = reinterpret_cast<uint64_t *>(d_sample_keys_->ptr());
  int *d_sampleidx2row =
      reinterpret_cast<int *>(d_sampleidx2rows_[cur_sampleidx2row_]->ptr());
  int *d_tmp_sampleidx2row =
      reinterpret_cast<int *>(d_sampleidx2rows_[1 - cur_sampleidx2row_]->ptr());

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                           temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1,
                                           len,
                                           sample_stream_));
  auto d_temp_storage = memory::Alloc(
      place_,
      temp_storage_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                           temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1,
                                           len,
                                           sample_stream_));

  cudaStreamSynchronize(sample_stream_);

  if (step == 1) {
    GraphFillFirstStepKernel<<<GET_BLOCKS(len),
                               CUDA_NUM_THREADS,
                               0,
                               sample_stream_>>>(d_prefix_sum,
                                                 d_tmp_sampleidx2row,
                                                 walk,
                                                 walk_ntype,
                                                 d_start_ids,
                                                 edge_src_id,
                                                 edge_dst_id,
                                                 len,
                                                 walk_degree_,
                                                 walk_len_,
                                                 d_actual_sample_size,
                                                 d_neighbors,
                                                 d_sample_keys);

  } else {
    GraphFillSampleKeysKernel<<<GET_BLOCKS(len),
                                CUDA_NUM_THREADS,
                                0,
                                sample_stream_>>>(d_neighbors,
                                                  d_sample_keys,
                                                  d_prefix_sum,
                                                  d_sampleidx2row,
                                                  d_tmp_sampleidx2row,
                                                  d_actual_sample_size,
                                                  cur_degree,
                                                  len);

    GraphDoWalkKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, sample_stream_>>>(
        d_neighbors,
        walk,
        walk_ntype,
        d_prefix_sum,
        d_actual_sample_size,
        cur_degree,
        step,
        len,
        len_per_row,
        d_tmp_sampleidx2row,
        walk_len_,
        edge_dst_id);
  }
  if (debug_mode_) {
    size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
    int *h_prefix_sum = new int[len + 1];
    int *h_actual_size = new int[len];
    int *h_offset2idx = new int[once_max_sample_keynum];
    cudaMemcpy(h_offset2idx,
               d_tmp_sampleidx2row,
               once_max_sample_keynum * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_prefix_sum,
               d_prefix_sum,
               (len + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < once_max_sample_keynum; xx++) {
      VLOG(2) << "h_offset2idx[" << xx << "]: " << h_offset2idx[xx];
    }
    for (int xx = 0; xx < len + 1; xx++) {
      VLOG(2) << "h_prefix_sum[" << xx << "]: " << h_prefix_sum[xx];
    }
    delete[] h_prefix_sum;
    delete[] h_actual_size;
    delete[] h_offset2idx;
  }
  cudaStreamSynchronize(sample_stream_);
  cur_sampleidx2row_ = 1 - cur_sampleidx2row_;
}

int GraphDataGenerator::FillSlotFeature(uint64_t *d_walk, size_t key_num) {
  platform::CUDADeviceGuard guard(gpuid_);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  std::shared_ptr<phi::Allocation> d_feature_list;
  std::shared_ptr<phi::Allocation> d_slot_list;

  if (sage_mode_) {
    size_t temp_storage_bytes = (key_num + 1) * sizeof(uint32_t);
    if (d_feature_size_list_buf_ == NULL ||
        d_feature_size_list_buf_->size() < temp_storage_bytes) {
      d_feature_size_list_buf_ =
          memory::AllocShared(this->place_, temp_storage_bytes);
    }
    if (d_feature_size_prefixsum_buf_ == NULL ||
        d_feature_size_prefixsum_buf_->size() < temp_storage_bytes) {
      d_feature_size_prefixsum_buf_ =
          memory::AllocShared(this->place_, temp_storage_bytes);
    }
  }

  uint32_t *d_feature_size_list_ptr =
      reinterpret_cast<uint32_t *>(d_feature_size_list_buf_->ptr());
  uint32_t *d_feature_size_prefixsum_ptr =
      reinterpret_cast<uint32_t *>(d_feature_size_prefixsum_buf_->ptr());
  int fea_num =
      gpu_graph_ptr->get_feature_info_of_nodes(gpuid_,
                                               d_walk,
                                               key_num,
                                               d_feature_size_list_ptr,
                                               d_feature_size_prefixsum_ptr,
                                               d_feature_list,
                                               d_slot_list);
  int64_t *slot_tensor_ptr_[slot_num_];
  int64_t *slot_lod_tensor_ptr_[slot_num_];
  if (fea_num == 0) {
    int64_t default_lod = 1;
    for (int i = 0; i < slot_num_; ++i) {
      slot_lod_tensor_ptr_[i] = feed_vec_[3 + 2 * i + 1]->mutable_data<int64_t>(
          {(long)key_num + 1}, this->place_);  // NOLINT
      slot_tensor_ptr_[i] =
          feed_vec_[3 + 2 * i]->mutable_data<int64_t>({1, 1}, this->place_);
      CUDA_CHECK(cudaMemsetAsync(
          slot_tensor_ptr_[i], 0, sizeof(int64_t), train_stream_));
      CUDA_CHECK(cudaMemsetAsync(slot_lod_tensor_ptr_[i],
                                 0,
                                 sizeof(int64_t) * key_num,
                                 train_stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<char *>(slot_lod_tensor_ptr_[i] + key_num),
          &default_lod,
          sizeof(int64_t),
          cudaMemcpyHostToDevice,
          train_stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    return 0;
  }

  uint64_t *d_feature_list_ptr =
      reinterpret_cast<uint64_t *>(d_feature_list->ptr());
  uint8_t *d_slot_list_ptr = reinterpret_cast<uint8_t *>(d_slot_list->ptr());

  std::shared_ptr<phi::Allocation> d_each_ins_slot_num_inner_prefix =
      memory::AllocShared(place_, (slot_num_ * key_num) * sizeof(uint32_t));
  std::shared_ptr<phi::Allocation> d_each_ins_slot_num =
      memory::AllocShared(place_, (slot_num_ * key_num) * sizeof(uint32_t));
  uint32_t *d_each_ins_slot_num_ptr =
      reinterpret_cast<uint32_t *>(d_each_ins_slot_num->ptr());
  uint32_t *d_each_ins_slot_num_inner_prefix_ptr =
      reinterpret_cast<uint32_t *>(d_each_ins_slot_num_inner_prefix->ptr());
  CUDA_CHECK(cudaMemsetAsync(d_each_ins_slot_num_ptr,
                             0,
                             slot_num_ * key_num * sizeof(uint32_t),
                             train_stream_));

  dim3 grid((key_num - 1) / 256 + 1);
  dim3 block(1, 256);

  get_each_ins_info<<<grid, block, 0, train_stream_>>>(
      d_slot_list_ptr,
      d_feature_size_list_ptr,
      d_feature_size_prefixsum_ptr,
      d_each_ins_slot_num_ptr,
      d_each_ins_slot_num_inner_prefix_ptr,
      key_num,
      slot_num_);

  std::vector<std::shared_ptr<phi::Allocation>> ins_slot_num(slot_num_,
                                                             nullptr);
  std::vector<uint64_t *> ins_slot_num_vecotr(slot_num_, NULL);
  std::shared_ptr<phi::Allocation> d_ins_slot_num_vector =
      memory::AllocShared(place_, (slot_num_) * sizeof(uint64_t *));
  uint64_t **d_ins_slot_num_vector_ptr =
      reinterpret_cast<uint64_t **>(d_ins_slot_num_vector->ptr());
  for (int i = 0; i < slot_num_; i++) {
    ins_slot_num[i] = memory::AllocShared(place_, key_num * sizeof(uint64_t));
    ins_slot_num_vecotr[i] =
        reinterpret_cast<uint64_t *>(ins_slot_num[i]->ptr());
  }
  CUDA_CHECK(
      cudaMemcpyAsync(reinterpret_cast<char *>(d_ins_slot_num_vector_ptr),
                      ins_slot_num_vecotr.data(),
                      sizeof(uint64_t *) * slot_num_,
                      cudaMemcpyHostToDevice,
                      train_stream_));
  fill_slot_num<<<grid, block, 0, train_stream_>>>(
      d_each_ins_slot_num_ptr, d_ins_slot_num_vector_ptr, key_num, slot_num_);
  CUDA_CHECK(cudaStreamSynchronize(train_stream_));

  for (int i = 0; i < slot_num_; ++i) {
    slot_lod_tensor_ptr_[i] = feed_vec_[3 + 2 * i + 1]->mutable_data<int64_t>(
        {(long)key_num + 1}, this->place_);  // NOLINT
  }
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                           temp_storage_bytes,
                                           ins_slot_num_vecotr[0],
                                           slot_lod_tensor_ptr_[0] + 1,
                                           key_num,
                                           train_stream_));
  CUDA_CHECK(cudaStreamSynchronize(train_stream_));
  auto d_temp_storage = memory::Alloc(
      this->place_,
      temp_storage_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(train_stream_)));
  std::vector<int64_t> each_slot_fea_num(slot_num_, 0);
  for (int i = 0; i < slot_num_; ++i) {
    CUDA_CHECK(cudaMemsetAsync(
        slot_lod_tensor_ptr_[i], 0, sizeof(uint64_t), train_stream_));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                             temp_storage_bytes,
                                             ins_slot_num_vecotr[i],
                                             slot_lod_tensor_ptr_[i] + 1,
                                             key_num,
                                             train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(&each_slot_fea_num[i],
                               slot_lod_tensor_ptr_[i] + key_num,
                               sizeof(uint64_t),
                               cudaMemcpyDeviceToHost,
                               train_stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(train_stream_));
  for (int i = 0; i < slot_num_; ++i) {
    slot_tensor_ptr_[i] = feed_vec_[3 + 2 * i]->mutable_data<int64_t>(
        {each_slot_fea_num[i], 1}, this->place_);
  }
  int64_t default_lod = 1;
  for (int i = 0; i < slot_num_; ++i) {
    fill_slot_tensor<<<grid, block, 0, train_stream_>>>(
        d_feature_list_ptr,
        d_feature_size_prefixsum_ptr,
        d_each_ins_slot_num_inner_prefix_ptr,
        ins_slot_num_vecotr[i],
        slot_lod_tensor_ptr_[i],
        slot_tensor_ptr_[i],
        i,
        slot_num_,
        key_num);
    // trick for empty tensor
    if (each_slot_fea_num[i] == 0) {
      slot_tensor_ptr_[i] =
          feed_vec_[3 + 2 * i]->mutable_data<int64_t>({1, 1}, this->place_);
      CUDA_CHECK(cudaMemsetAsync(
          slot_tensor_ptr_[i], 0, sizeof(uint64_t), train_stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<char *>(slot_lod_tensor_ptr_[i] + key_num),
          &default_lod,
          sizeof(int64_t),
          cudaMemcpyHostToDevice,
          train_stream_));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(train_stream_));

  if (debug_mode_) {
    std::vector<uint32_t> h_feature_size_list(key_num, 0);
    std::vector<uint32_t> h_feature_size_list_prefixsum(key_num, 0);
    std::vector<uint64_t> node_list(key_num, 0);
    std::vector<uint64_t> h_feature_list(fea_num, 0);
    std::vector<uint8_t> h_slot_list(fea_num, 0);

    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<char *>(h_feature_size_list.data()),
                        d_feature_size_list_ptr,
                        sizeof(uint32_t) * key_num,
                        cudaMemcpyDeviceToHost,
                        train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char *>(h_feature_size_list_prefixsum.data()),
        d_feature_size_prefixsum_ptr,
        sizeof(uint32_t) * key_num,
        cudaMemcpyDeviceToHost,
        train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(node_list.data()),
                               d_walk,
                               sizeof(uint64_t) * key_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_feature_list.data()),
                               d_feature_list_ptr,
                               sizeof(uint64_t) * fea_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_list.data()),
                               d_slot_list_ptr,
                               sizeof(uint8_t) * fea_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));

    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    for (size_t i = 0; i < key_num; i++) {
      std::stringstream ss;
      ss << "node_id: " << node_list[i]
         << " fea_num: " << h_feature_size_list[i] << " offset "
         << h_feature_size_list_prefixsum[i] << " slot: ";
      for (uint32_t j = 0; j < h_feature_size_list[i]; j++) {
        ss << int(h_slot_list[h_feature_size_list_prefixsum[i] + j]) << " : "
           << h_feature_list[h_feature_size_list_prefixsum[i] + j] << "  ";
      }
      VLOG(0) << ss.str();
    }
    VLOG(0) << "all fea_num is " << fea_num << " calc fea_num is "
            << h_feature_size_list[key_num - 1] +
                   h_feature_size_list_prefixsum[key_num - 1];
    for (int i = 0; i < slot_num_; ++i) {
      std::vector<int64_t> h_slot_lod_tensor(key_num + 1, 0);
      CUDA_CHECK(
          cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_lod_tensor.data()),
                          slot_lod_tensor_ptr_[i],
                          sizeof(int64_t) * (key_num + 1),
                          cudaMemcpyDeviceToHost,
                          train_stream_));
      CUDA_CHECK(cudaStreamSynchronize(train_stream_));
      std::stringstream ss_lod;
      std::stringstream ss_tensor;
      ss_lod << " slot " << i << " lod is [";
      for (size_t j = 0; j < key_num + 1; j++) {
        ss_lod << h_slot_lod_tensor[j] << ",";
      }
      ss_lod << "]";
      std::vector<int64_t> h_slot_tensor(h_slot_lod_tensor[key_num], 0);
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_tensor.data()),
                                 slot_tensor_ptr_[i],
                                 sizeof(int64_t) * h_slot_lod_tensor[key_num],
                                 cudaMemcpyDeviceToHost,
                                 train_stream_));
      CUDA_CHECK(cudaStreamSynchronize(train_stream_));

      ss_tensor << " tensor is [ ";
      for (size_t j = 0; j < h_slot_lod_tensor[key_num]; j++) {
        ss_tensor << h_slot_tensor[j] << ",";
      }
      ss_tensor << "]";
      VLOG(0) << ss_lod.str() << "  " << ss_tensor.str();
    }
  }

  return 0;
}

int GraphDataGenerator::FillFeatureBuf(uint64_t *d_walk,
                                       uint64_t *d_feature,
                                       size_t key_num) {
  platform::CUDADeviceGuard guard(gpuid_);

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  int ret = gpu_graph_ptr->get_feature_of_nodes(
      gpuid_,
      d_walk,
      d_feature,
      key_num,
      slot_num_,
      reinterpret_cast<int *>(d_slot_feature_num_map_->ptr()),
      fea_num_per_node_);
  return ret;
}

int GraphDataGenerator::FillFeatureBuf(
    std::shared_ptr<phi::Allocation> d_walk,
    std::shared_ptr<phi::Allocation> d_feature) {
  platform::CUDADeviceGuard guard(gpuid_);

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  int ret = gpu_graph_ptr->get_feature_of_nodes(
      gpuid_,
      reinterpret_cast<uint64_t *>(d_walk->ptr()),
      reinterpret_cast<uint64_t *>(d_feature->ptr()),
      buf_size_,
      slot_num_,
      reinterpret_cast<int *>(d_slot_feature_num_map_->ptr()),
      fea_num_per_node_);
  return ret;
}

// 对于deepwalk模式，尝试插入table，0表示插入成功，1表示插入失败；
// 对于sage模式，尝试插入table，table数量不够则清空table重新插入，返回值无影响。
int GraphDataGenerator::InsertTable(
    const uint64_t *d_keys,
    uint64_t len,
    std::shared_ptr<phi::Allocation> d_uniq_node_num) {
  // Used under NOT WHOLE_HBM.
  uint64_t h_uniq_node_num = 0;
  uint64_t *d_uniq_node_num_ptr =
      reinterpret_cast<uint64_t *>(d_uniq_node_num->ptr());
  cudaMemcpyAsync(&h_uniq_node_num,
                  d_uniq_node_num_ptr,
                  sizeof(uint64_t),
                  cudaMemcpyDeviceToHost,
                  sample_stream_);
  cudaStreamSynchronize(sample_stream_);

  if (gpu_graph_training_) {
    VLOG(2) << "table capacity: " << train_table_cap_ << ", " << h_uniq_node_num
            << " used";
    if (h_uniq_node_num + len >= train_table_cap_) {
      if (!sage_mode_) {
        return 1;
      } else {
        // Copy unique nodes first.
        uint64_t copy_len = CopyUniqueNodes();
        copy_unique_len_ += copy_len;
        table_->clear(sample_stream_);
        cudaMemsetAsync(
            d_uniq_node_num_ptr, 0, sizeof(uint64_t), sample_stream_);
      }
    }
  } else {
    // used only for sage_mode.
    if (h_uniq_node_num + len >= infer_table_cap_) {
      uint64_t copy_len = CopyUniqueNodes();
      copy_unique_len_ += copy_len;
      table_->clear(sample_stream_);
      cudaMemsetAsync(d_uniq_node_num_ptr, 0, sizeof(uint64_t), sample_stream_);
    }
  }

  table_->insert(d_keys, len, d_uniq_node_num_ptr, sample_stream_);
  CUDA_CHECK(cudaStreamSynchronize(sample_stream_));
  return 0;
}

std::vector<std::shared_ptr<phi::Allocation>>
GraphDataGenerator::SampleNeighbors(int64_t *uniq_nodes,
                                    int len,
                                    int sample_size,
                                    std::vector<int> &edges_split_num,
                                    int64_t *neighbor_len) {
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto sample_res = gpu_graph_ptr->graph_neighbor_sample_all_edge_type(
      gpuid_,
      edge_to_id_len_,
      reinterpret_cast<uint64_t *>(uniq_nodes),
      sample_size,
      len,
      edge_type_graph_);

  int *all_sample_count_ptr =
      reinterpret_cast<int *>(sample_res.actual_sample_size_mem->ptr());

  auto cumsum_actual_sample_size = memory::Alloc(
      place_,
      (len * edge_to_id_len_ + 1) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int *cumsum_actual_sample_size_ptr =
      reinterpret_cast<int *>(cumsum_actual_sample_size->ptr());
  cudaMemsetAsync(cumsum_actual_sample_size_ptr,
                  0,
                  (len * edge_to_id_len_ + 1) * sizeof(int),
                  sample_stream_);

  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                           temp_storage_bytes,
                                           all_sample_count_ptr,
                                           cumsum_actual_sample_size_ptr + 1,
                                           len * edge_to_id_len_,
                                           sample_stream_));
  auto d_temp_storage = memory::Alloc(
      place_,
      temp_storage_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                           temp_storage_bytes,
                                           all_sample_count_ptr,
                                           cumsum_actual_sample_size_ptr + 1,
                                           len * edge_to_id_len_,
                                           sample_stream_));
  cudaStreamSynchronize(sample_stream_);

  edges_split_num.resize(edge_to_id_len_);
  for (int i = 0; i < edge_to_id_len_; i++) {
    cudaMemcpyAsync(edges_split_num.data() + i,
                    cumsum_actual_sample_size_ptr + (i + 1) * len,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    sample_stream_);
  }

  CUDA_CHECK(cudaStreamSynchronize(sample_stream_));

  int all_sample_size = edges_split_num[edge_to_id_len_ - 1];
  auto final_sample_val = memory::AllocShared(
      place_,
      all_sample_size * sizeof(int64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  auto final_sample_val_dst = memory::AllocShared(
      place_,
      all_sample_size * sizeof(int64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int64_t *final_sample_val_ptr =
      reinterpret_cast<int64_t *>(final_sample_val->ptr());
  int64_t *final_sample_val_dst_ptr =
      reinterpret_cast<int64_t *>(final_sample_val_dst->ptr());
  int64_t *all_sample_val_ptr =
      reinterpret_cast<int64_t *>(sample_res.val_mem->ptr());
  FillActualNeighbors<<<GET_BLOCKS(len * edge_to_id_len_),
                        CUDA_NUM_THREADS,
                        0,
                        sample_stream_>>>(all_sample_val_ptr,
                                          final_sample_val_ptr,
                                          final_sample_val_dst_ptr,
                                          all_sample_count_ptr,
                                          cumsum_actual_sample_size_ptr,
                                          sample_size,
                                          len * edge_to_id_len_,
                                          len);
  *neighbor_len = all_sample_size;
  cudaStreamSynchronize(sample_stream_);

  std::vector<std::shared_ptr<phi::Allocation>> sample_results;
  sample_results.emplace_back(final_sample_val);
  sample_results.emplace_back(final_sample_val_dst);
  return sample_results;
}

std::shared_ptr<phi::Allocation> GraphDataGenerator::FillReindexHashTable(
    int64_t *input,
    int num_input,
    int64_t len_hashtable,
    int64_t *keys,
    int *values,
    int *key_index,
    int *final_nodes_len) {
  phi::BuildHashTable<int64_t>
      <<<GET_BLOCKS(num_input), CUDA_NUM_THREADS, 0, sample_stream_>>>(
          input, num_input, len_hashtable, keys, key_index);

  // Get item index count.
  auto item_count = memory::Alloc(
      place_,
      (num_input + 1) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int *item_count_ptr = reinterpret_cast<int *>(item_count->ptr());
  cudaMemsetAsync(
      item_count_ptr, 0, sizeof(int) * (num_input + 1), sample_stream_);
  phi::GetItemIndexCount<int64_t>
      <<<GET_BLOCKS(num_input), CUDA_NUM_THREADS, 0, sample_stream_>>>(
          input, item_count_ptr, num_input, len_hashtable, keys, key_index);

  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(NULL,
                                temp_storage_bytes,
                                item_count_ptr,
                                item_count_ptr,
                                num_input + 1,
                                sample_stream_);
  auto d_temp_storage = memory::Alloc(
      place_,
      temp_storage_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  cub::DeviceScan::ExclusiveSum(d_temp_storage->ptr(),
                                temp_storage_bytes,
                                item_count_ptr,
                                item_count_ptr,
                                num_input + 1,
                                sample_stream_);

  int total_unique_items = 0;
  cudaMemcpyAsync(&total_unique_items,
                  item_count_ptr + num_input,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  sample_stream_);
  cudaStreamSynchronize(sample_stream_);

  auto unique_items = memory::AllocShared(
      place_,
      total_unique_items * sizeof(int64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int64_t *unique_items_ptr = reinterpret_cast<int64_t *>(unique_items->ptr());
  *final_nodes_len = total_unique_items;

  // Get unique items
  phi::FillUniqueItems<int64_t>
      <<<GET_BLOCKS(num_input), CUDA_NUM_THREADS, 0, sample_stream_>>>(
          input,
          num_input,
          len_hashtable,
          unique_items_ptr,
          item_count_ptr,
          keys,
          values,
          key_index);
  cudaStreamSynchronize(sample_stream_);
  return unique_items;
}

std::shared_ptr<phi::Allocation> GraphDataGenerator::GetReindexResult(
    int64_t *reindex_src_data,
    int64_t *center_nodes,
    int *final_nodes_len,
    int node_len,
    int64_t neighbor_len) {
  // Reset reindex table
  int64_t *d_reindex_table_key_ptr =
      reinterpret_cast<int64_t *>(d_reindex_table_key_->ptr());
  int *d_reindex_table_value_ptr =
      reinterpret_cast<int *>(d_reindex_table_value_->ptr());
  int *d_reindex_table_index_ptr =
      reinterpret_cast<int *>(d_reindex_table_index_->ptr());

  // Fill table with -1.
  cudaMemsetAsync(d_reindex_table_key_ptr,
                  -1,
                  reindex_table_size_ * sizeof(int64_t),
                  sample_stream_);
  cudaMemsetAsync(d_reindex_table_value_ptr,
                  -1,
                  reindex_table_size_ * sizeof(int),
                  sample_stream_);
  cudaMemsetAsync(d_reindex_table_index_ptr,
                  -1,
                  reindex_table_size_ * sizeof(int),
                  sample_stream_);

  auto all_nodes = memory::AllocShared(
      place_,
      (node_len + neighbor_len) * sizeof(int64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int64_t *all_nodes_data = reinterpret_cast<int64_t *>(all_nodes->ptr());

  cudaMemcpyAsync(all_nodes_data,
                  center_nodes,
                  sizeof(int64_t) * node_len,
                  cudaMemcpyDeviceToDevice,
                  sample_stream_);
  cudaMemcpyAsync(all_nodes_data + node_len,
                  reindex_src_data,
                  sizeof(int64_t) * neighbor_len,
                  cudaMemcpyDeviceToDevice,
                  sample_stream_);

  cudaStreamSynchronize(sample_stream_);

  auto final_nodes = FillReindexHashTable(all_nodes_data,
                                          node_len + neighbor_len,
                                          reindex_table_size_,
                                          d_reindex_table_key_ptr,
                                          d_reindex_table_value_ptr,
                                          d_reindex_table_index_ptr,
                                          final_nodes_len);

  phi::ReindexSrcOutput<int64_t>
      <<<GET_BLOCKS(neighbor_len), CUDA_NUM_THREADS, 0, sample_stream_>>>(
          reindex_src_data,
          neighbor_len,
          reindex_table_size_,
          d_reindex_table_key_ptr,
          d_reindex_table_value_ptr);
  return final_nodes;
}

std::shared_ptr<phi::Allocation> GraphDataGenerator::GenerateSampleGraph(
    uint64_t *node_ids,
    int len,
    int *final_len,
    std::shared_ptr<phi::Allocation> &inverse) {
  VLOG(2) << "Get Unique Nodes";

  auto uniq_nodes = memory::Alloc(
      place_,
      len * sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int *inverse_ptr = reinterpret_cast<int *>(inverse->ptr());
  int64_t *uniq_nodes_data = reinterpret_cast<int64_t *>(uniq_nodes->ptr());
  int uniq_len = dedup_keys_and_fillidx(
      len,
      node_ids,
      reinterpret_cast<uint64_t *>(uniq_nodes_data),
      reinterpret_cast<uint64_t *>(d_sorted_keys_->ptr()),
      reinterpret_cast<uint32_t *>(inverse_ptr),
      reinterpret_cast<uint32_t *>(d_sorted_idx_->ptr()),
      reinterpret_cast<uint32_t *>(d_offset_->ptr()),
      reinterpret_cast<uint32_t *>(d_merged_cnts_->ptr()),
      sample_stream_,
      d_buf_,
      place_);
  int len_samples = samples_.size();

  VLOG(2) << "Sample Neighbors and Reindex";
  std::vector<int> edges_split_num;
  std::vector<std::shared_ptr<phi::Allocation>> final_nodes_vec;
  std::vector<std::shared_ptr<phi::Allocation>> graph_edges;
  std::vector<std::vector<int>> edges_split_num_for_graph;
  std::vector<int> final_nodes_len_vec;

  for (int i = 0; i < len_samples; i++) {
    edges_split_num.clear();
    std::shared_ptr<phi::Allocation> neighbors, reindex_dst;
    int64_t neighbors_len = 0;
    if (i == 0) {
      auto sample_results = SampleNeighbors(uniq_nodes_data,
                                            uniq_len,
                                            samples_[i],
                                            edges_split_num,
                                            &neighbors_len);
      neighbors = sample_results[0];
      reindex_dst = sample_results[1];
      edges_split_num.push_back(uniq_len);
    } else {
      int64_t *final_nodes_data =
          reinterpret_cast<int64_t *>(final_nodes_vec[i - 1]->ptr());
      auto sample_results = SampleNeighbors(final_nodes_data,
                                            final_nodes_len_vec[i - 1],
                                            samples_[i],
                                            edges_split_num,
                                            &neighbors_len);
      neighbors = sample_results[0];
      reindex_dst = sample_results[1];
      edges_split_num.push_back(final_nodes_len_vec[i - 1]);
    }

    int64_t *reindex_src_data = reinterpret_cast<int64_t *>(neighbors->ptr());
    int final_nodes_len = 0;
    if (i == 0) {
      auto tmp_final_nodes = GetReindexResult(reindex_src_data,
                                              uniq_nodes_data,
                                              &final_nodes_len,
                                              uniq_len,
                                              neighbors_len);
      final_nodes_vec.emplace_back(tmp_final_nodes);
      final_nodes_len_vec.emplace_back(final_nodes_len);
    } else {
      int64_t *final_nodes_data =
          reinterpret_cast<int64_t *>(final_nodes_vec[i - 1]->ptr());
      auto tmp_final_nodes = GetReindexResult(reindex_src_data,
                                              final_nodes_data,
                                              &final_nodes_len,
                                              final_nodes_len_vec[i - 1],
                                              neighbors_len);
      final_nodes_vec.emplace_back(tmp_final_nodes);
      final_nodes_len_vec.emplace_back(final_nodes_len);
    }
    edges_split_num.emplace_back(
        final_nodes_len_vec[i]);  // [edges_split_num, next_num_nodes,
                                  // num_nodes]
    edges_split_num.emplace_back(neighbors_len);
    graph_edges.emplace_back(neighbors);
    graph_edges.emplace_back(reindex_dst);
    edges_split_num_for_graph.emplace_back(edges_split_num);
  }
  graph_edges_vec_.emplace_back(graph_edges);
  edges_split_num_vec_.emplace_back(edges_split_num_for_graph);

  *final_len = final_nodes_len_vec[len_samples - 1];
  return final_nodes_vec[len_samples - 1];
}

std::shared_ptr<phi::Allocation> GraphDataGenerator::GetNodeDegree(
    uint64_t *node_ids, int len) {
  auto node_degree = memory::AllocShared(
      place_,
      len * edge_to_id_len_ * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto edge_to_id = gpu_graph_ptr->edge_to_id;
  for (auto &iter : edge_to_id) {
    int edge_idx = iter.second;
    gpu_graph_ptr->get_node_degree(
        gpuid_, edge_idx, node_ids, len, node_degree);
  }
  return node_degree;
}

uint64_t GraphDataGenerator::CopyUniqueNodes() {
  if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
    uint64_t h_uniq_node_num = 0;
    uint64_t *d_uniq_node_num =
        reinterpret_cast<uint64_t *>(d_uniq_node_num_->ptr());
    cudaMemcpyAsync(&h_uniq_node_num,
                    d_uniq_node_num,
                    sizeof(uint64_t),
                    cudaMemcpyDeviceToHost,
                    sample_stream_);
    cudaStreamSynchronize(sample_stream_);
    auto d_uniq_node = memory::AllocShared(
        place_,
        h_uniq_node_num * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    uint64_t *d_uniq_node_ptr =
        reinterpret_cast<uint64_t *>(d_uniq_node->ptr());

    auto d_node_cursor = memory::AllocShared(
        place_,
        sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));

    uint64_t *d_node_cursor_ptr =
        reinterpret_cast<uint64_t *>(d_node_cursor->ptr());
    cudaMemsetAsync(d_node_cursor_ptr, 0, sizeof(uint64_t), sample_stream_);
    // uint64_t unused_key = std::numeric_limits<uint64_t>::max();
    table_->get_keys(d_uniq_node_ptr, d_node_cursor_ptr, sample_stream_);

    cudaStreamSynchronize(sample_stream_);

    host_vec_.resize(h_uniq_node_num + copy_unique_len_);
    cudaMemcpyAsync(host_vec_.data() + copy_unique_len_,
                    d_uniq_node_ptr,
                    sizeof(uint64_t) * h_uniq_node_num,
                    cudaMemcpyDeviceToHost,
                    sample_stream_);
    cudaStreamSynchronize(sample_stream_);
    return h_uniq_node_num;
  }
  return 0;
}

void GraphDataGenerator::DoWalkandSage() {
  int device_id = place_.GetDeviceId();
  debug_gpu_memory_info(device_id, "DoWalkandSage start");
  platform::CUDADeviceGuard guard(gpuid_);
  if (gpu_graph_training_) {
    // train
    bool train_flag;
    if (FLAGS_graph_metapath_split_opt) {
      train_flag = FillWalkBufMultiPath();
    } else {
      train_flag = FillWalkBuf();
    }

    if (sage_mode_) {
      sage_batch_num_ = 0;
      if (train_flag) {
        int total_instance = 0, uniq_instance = 0;
        bool ins_pair_flag = true;
        uint64_t *ins_buf, *ins_cursor;
        while (ins_pair_flag) {
          int res = 0;
          while (ins_buf_pair_len_ < batch_size_) {
            res = FillInsBuf(sample_stream_);
            if (res == -1) {
              if (ins_buf_pair_len_ == 0) {
                ins_pair_flag = false;
              }
              break;
            }
          }

          if (!ins_pair_flag) {
            break;
          }

          total_instance =
              ins_buf_pair_len_ < batch_size_ ? ins_buf_pair_len_ : batch_size_;
          total_instance *= 2;

          ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
          ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
          auto inverse = memory::AllocShared(
              place_,
              total_instance * sizeof(int),
              phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
          auto final_sage_nodes = GenerateSampleGraph(
              ins_cursor, total_instance, &uniq_instance, inverse);
          uint64_t *final_sage_nodes_ptr =
              reinterpret_cast<uint64_t *>(final_sage_nodes->ptr());
          if (get_degree_) {
            auto node_degrees =
                GetNodeDegree(final_sage_nodes_ptr, uniq_instance);
            node_degree_vec_.emplace_back(node_degrees);
          }
          cudaStreamSynchronize(sample_stream_);
          if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
            uint64_t *final_sage_nodes_ptr =
                reinterpret_cast<uint64_t *>(final_sage_nodes->ptr());
            InsertTable(final_sage_nodes_ptr, uniq_instance, d_uniq_node_num_);
          }
          final_sage_nodes_vec_.emplace_back(final_sage_nodes);
          inverse_vec_.emplace_back(inverse);
          uniq_instance_vec_.emplace_back(uniq_instance);
          total_instance_vec_.emplace_back(total_instance);
          ins_buf_pair_len_ -= total_instance / 2;
          sage_batch_num_ += 1;
        }
        uint64_t h_uniq_node_num = CopyUniqueNodes();
        VLOG(1) << "train sage_batch_num: " << sage_batch_num_;
      }
    }
  } else {
    // infer
    bool infer_flag = FillInferBuf();
    if (sage_mode_) {
      sage_batch_num_ = 0;
      if (infer_flag) {
        int total_instance = 0, uniq_instance = 0;
        total_instance = (infer_node_start_ + batch_size_ <= infer_node_end_)
                             ? batch_size_
                             : infer_node_end_ - infer_node_start_;
        total_instance *= 2;
        while (total_instance != 0) {
          uint64_t *d_type_keys =
              reinterpret_cast<uint64_t *>(d_device_keys_[cursor_]->ptr());
          d_type_keys += infer_node_start_;
          infer_node_start_ += total_instance / 2;
          auto node_buf = memory::AllocShared(
              place_,
              total_instance * sizeof(uint64_t),
              phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
          int64_t *node_buf_ptr = reinterpret_cast<int64_t *>(node_buf->ptr());
          CopyDuplicateKeys<<<GET_BLOCKS(total_instance / 2),
                              CUDA_NUM_THREADS,
                              0,
                              sample_stream_>>>(
              node_buf_ptr, d_type_keys, total_instance / 2);
          uint64_t *node_buf_ptr_ =
              reinterpret_cast<uint64_t *>(node_buf->ptr());
          auto inverse = memory::AllocShared(
              place_,
              total_instance * sizeof(int),
              phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
          auto final_sage_nodes = GenerateSampleGraph(
              node_buf_ptr_, total_instance, &uniq_instance, inverse);
          uint64_t *final_sage_nodes_ptr =
              reinterpret_cast<uint64_t *>(final_sage_nodes->ptr());
          if (get_degree_) {
            auto node_degrees =
                GetNodeDegree(final_sage_nodes_ptr, uniq_instance);
            node_degree_vec_.emplace_back(node_degrees);
          }
          cudaStreamSynchronize(sample_stream_);
          if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
            uint64_t *final_sage_nodes_ptr =
                reinterpret_cast<uint64_t *>(final_sage_nodes->ptr());
            InsertTable(final_sage_nodes_ptr, uniq_instance, d_uniq_node_num_);
          }
          final_sage_nodes_vec_.emplace_back(final_sage_nodes);
          inverse_vec_.emplace_back(inverse);
          uniq_instance_vec_.emplace_back(uniq_instance);
          total_instance_vec_.emplace_back(total_instance);
          sage_batch_num_ += 1;

          total_instance = (infer_node_start_ + batch_size_ <= infer_node_end_)
                               ? batch_size_
                               : infer_node_end_ - infer_node_start_;
          total_instance *= 2;
        }

        uint64_t h_uniq_node_num = CopyUniqueNodes();
        VLOG(1) << "infer sage_batch_num: " << sage_batch_num_;
      }
    }
  }
  debug_gpu_memory_info(device_id, "DoWalkandSage end");
}

void GraphDataGenerator::clear_gpu_mem() {
  d_len_per_row_.reset();
  d_sample_keys_.reset();
  d_prefix_sum_.reset();
  for (size_t i = 0; i < d_sampleidx2rows_.size(); i++) {
    d_sampleidx2rows_[i].reset();
  }
  delete table_;
  if (sage_mode_) {
    d_reindex_table_key_.reset();
    d_reindex_table_value_.reset();
    d_reindex_table_index_.reset();
    d_sorted_keys_.reset();
    d_sorted_idx_.reset();
    d_offset_.reset();
    d_merged_cnts_.reset();
  }
}

int GraphDataGenerator::FillInferBuf() {
  platform::CUDADeviceGuard guard(gpuid_);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto &global_infer_node_type_start =
      gpu_graph_ptr->global_infer_node_type_start_[gpuid_];
  auto &infer_cursor = gpu_graph_ptr->infer_cursor_[thread_id_];
  total_row_ = 0;
  if (infer_cursor < h_device_keys_len_.size()) {
    if (global_infer_node_type_start[infer_cursor] >=
        h_device_keys_len_[infer_cursor]) {
      infer_cursor++;
      if (infer_cursor >= h_device_keys_len_.size()) {
        return 0;
      }
    }
    if (!infer_node_type_index_set_.empty()) {
      while (infer_cursor < h_device_keys_len_.size()) {
        if (infer_node_type_index_set_.find(infer_cursor) ==
            infer_node_type_index_set_.end()) {
          VLOG(2) << "Skip cursor[" << infer_cursor << "]";
          infer_cursor++;
          continue;
        } else {
          VLOG(2) << "Not skip cursor[" << infer_cursor << "]";
          break;
        }
      }
      if (infer_cursor >= h_device_keys_len_.size()) {
        return 0;
      }
    }

    size_t device_key_size = h_device_keys_len_[infer_cursor];
    total_row_ =
        (global_infer_node_type_start[infer_cursor] + infer_table_cap_ <=
         device_key_size)
            ? infer_table_cap_
            : device_key_size - global_infer_node_type_start[infer_cursor];

    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys_[infer_cursor]->ptr());
    if (!sage_mode_) {
      host_vec_.resize(total_row_);
      cudaMemcpyAsync(host_vec_.data(),
                      d_type_keys + global_infer_node_type_start[infer_cursor],
                      sizeof(uint64_t) * total_row_,
                      cudaMemcpyDeviceToHost,
                      sample_stream_);
      cudaStreamSynchronize(sample_stream_);
    }
    VLOG(1) << "cursor: " << infer_cursor
            << " start: " << global_infer_node_type_start[infer_cursor]
            << " num: " << total_row_;
    infer_node_start_ = global_infer_node_type_start[infer_cursor];
    global_infer_node_type_start[infer_cursor] += total_row_;
    infer_node_end_ = global_infer_node_type_start[infer_cursor];
    cursor_ = infer_cursor;
  }
  return 1;
}

void GraphDataGenerator::ClearSampleState() {
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto &finish_node_type = gpu_graph_ptr->finish_node_type_[gpuid_];
  auto &node_type_start = gpu_graph_ptr->node_type_start_[gpuid_];
  finish_node_type.clear();
  for (auto iter = node_type_start.begin(); iter != node_type_start.end();
       iter++) {
    iter->second = 0;
  }
}

int GraphDataGenerator::FillWalkBuf() {
  platform::CUDADeviceGuard guard(gpuid_);
  size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
  ////////
  uint64_t *h_walk;
  uint64_t *h_sample_keys;
  int *h_offset2idx;
  int *h_len_per_row;
  uint64_t *h_prefix_sum;
  if (debug_mode_) {
    h_walk = new uint64_t[buf_size_];
    h_sample_keys = new uint64_t[once_max_sample_keynum];
    h_offset2idx = new int[once_max_sample_keynum];
    h_len_per_row = new int[once_max_sample_keynum];
    h_prefix_sum = new uint64_t[once_max_sample_keynum + 1];
  }
  ///////
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_->ptr());
  int *len_per_row = reinterpret_cast<int *>(d_len_per_row_->ptr());
  uint64_t *d_sample_keys = reinterpret_cast<uint64_t *>(d_sample_keys_->ptr());
  cudaMemsetAsync(walk, 0, buf_size_ * sizeof(uint64_t), sample_stream_);
  uint8_t *walk_ntype = NULL;
  if (excluded_train_pair_len_ > 0) {
    walk_ntype = reinterpret_cast<uint8_t *>(d_walk_ntype_->ptr());
    cudaMemsetAsync(walk_ntype, 0, buf_size_ * sizeof(uint8_t), sample_stream_);
  }
  // cudaMemsetAsync(
  //     len_per_row, 0, once_max_sample_keynum * sizeof(int), sample_stream_);
  int sample_times = 0;
  int i = 0;
  total_row_ = 0;

  // 获取全局采样状态
  auto &first_node_type = gpu_graph_ptr->first_node_type_;
  auto &meta_path = gpu_graph_ptr->meta_path_;
  auto &node_type_start = gpu_graph_ptr->node_type_start_[gpuid_];
  auto &finish_node_type = gpu_graph_ptr->finish_node_type_[gpuid_];
  auto &type_to_index = gpu_graph_ptr->get_graph_type_to_index();
  auto &cursor = gpu_graph_ptr->cursor_[thread_id_];
  size_t node_type_len = first_node_type.size();
  int remain_size =
      buf_size_ - walk_degree_ * once_sample_startid_len_ * walk_len_;
  int total_samples = 0;

  while (i <= remain_size) {
    int cur_node_idx = cursor % node_type_len;
    int node_type = first_node_type[cur_node_idx];
    auto &path = meta_path[cur_node_idx];
    size_t start = node_type_start[node_type];
    VLOG(2) << "cur_node_idx = " << cur_node_idx
            << " meta_path.size = " << meta_path.size();
    // auto node_query_result = gpu_graph_ptr->query_node_list(
    //     gpuid_, node_type, start, once_sample_startid_len_);

    // int tmp_len = node_query_result.actual_sample_size;
    VLOG(2) << "choose start type: " << node_type;
    int type_index = type_to_index[node_type];
    size_t device_key_size = h_device_keys_len_[type_index];
    VLOG(2) << "type: " << node_type << " size: " << device_key_size
            << " start: " << start;
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys_[type_index]->ptr());
    int tmp_len = start + once_sample_startid_len_ > device_key_size
                      ? device_key_size - start
                      : once_sample_startid_len_;
    bool update = true;
    if (tmp_len == 0) {
      finish_node_type.insert(node_type);
      if (finish_node_type.size() == node_type_start.size()) {
        cursor = 0;
        epoch_finish_ = true;
        break;
      }
      cursor += 1;
      continue;
    }

    VLOG(2) << "gpuid = " << gpuid_ << " path[0] = " << path[0];
    uint64_t *cur_walk = walk + i;
    uint8_t *cur_walk_ntype = NULL;
    if (excluded_train_pair_len_ > 0) {
      cur_walk_ntype = walk_ntype + i;
    }

    NeighborSampleQuery q;
    q.initialize(gpuid_,
                 path[0],
                 (uint64_t)(d_type_keys + start),
                 walk_degree_,
                 tmp_len);
    auto sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);

    int step = 1;
    VLOG(2) << "sample edge type: " << path[0] << " step: " << 1;
    jump_rows_ = sample_res.total_sample_size;
    total_samples += sample_res.total_sample_size;
    VLOG(2) << "i = " << i << " start = " << start << " tmp_len = " << tmp_len
            << " cursor = " << node_type << " cur_node_idx = " << cur_node_idx
            << " jump row: " << jump_rows_;
    VLOG(2) << "jump_row: " << jump_rows_;
    if (jump_rows_ == 0) {
      node_type_start[node_type] = tmp_len + start;
      cursor += 1;
      continue;
    }

    if (!sage_mode_) {
      if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
        if (InsertTable(d_type_keys + start, tmp_len, d_uniq_node_num_) != 0) {
          VLOG(2) << "in step 0, insert key stage, table is full";
          update = false;
          break;
        }
        if (InsertTable(sample_res.actual_val,
                        sample_res.total_sample_size,
                        d_uniq_node_num_) != 0) {
          VLOG(2) << "in step 0, insert sample res stage, table is full";
          update = false;
          break;
        }
      }
    }
    FillOneStep(d_type_keys + start,
                path[0],
                cur_walk,
                cur_walk_ntype,
                tmp_len,
                sample_res,
                walk_degree_,
                step,
                len_per_row);
    /////////
    if (debug_mode_) {
      cudaMemcpy(
          h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < buf_size_; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }

    VLOG(2) << "sample, step=" << step << " sample_keys=" << tmp_len
            << " sample_res_len=" << sample_res.total_sample_size;

    /////////
    step++;
    size_t path_len = path.size();
    for (; step < walk_len_; step++) {
      if (sample_res.total_sample_size == 0) {
        VLOG(2) << "sample finish, step=" << step;
        break;
      }
      auto sample_key_mem = sample_res.actual_val_mem;
      uint64_t *sample_keys_ptr =
          reinterpret_cast<uint64_t *>(sample_key_mem->ptr());
      int edge_type_id = path[(step - 1) % path_len];
      VLOG(2) << "sample edge type: " << edge_type_id << " step: " << step;
      q.initialize(gpuid_,
                   edge_type_id,
                   (uint64_t)sample_keys_ptr,
                   1,
                   sample_res.total_sample_size);
      int sample_key_len = sample_res.total_sample_size;
      sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);
      total_samples += sample_res.total_sample_size;
      if (!sage_mode_) {
        if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
          if (InsertTable(sample_res.actual_val,
                          sample_res.total_sample_size,
                          d_uniq_node_num_) != 0) {
            VLOG(2) << "in step: " << step << ", table is full";
            update = false;
            break;
          }
        }
      }
      FillOneStep(d_type_keys + start,
                  edge_type_id,
                  cur_walk,
                  cur_walk_ntype,
                  sample_key_len,
                  sample_res,
                  1,
                  step,
                  len_per_row);
      if (debug_mode_) {
        cudaMemcpy(
            h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        for (int xx = 0; xx < buf_size_; xx++) {
          VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
        }
      }

      VLOG(2) << "sample, step=" << step << " sample_keys=" << sample_key_len
              << " sample_res_len=" << sample_res.total_sample_size;
    }
    // 此时更新全局采样状态
    if (update == true) {
      node_type_start[node_type] = tmp_len + start;
      i += jump_rows_ * walk_len_;
      total_row_ += jump_rows_;
      cursor += 1;
      sample_times++;
    } else {
      VLOG(2) << "table is full, not update stat!";
      break;
    }
  }
  platform::CUDADeviceGuard guard2(gpuid_);
  buf_state_.Reset(total_row_);
  int *d_random_row = reinterpret_cast<int *>(d_random_row_->ptr());

  thrust::random::default_random_engine engine(shuffle_seed_);
  const auto &exec_policy = thrust::cuda::par.on(sample_stream_);
  thrust::counting_iterator<int> cnt_iter(0);
  thrust::shuffle_copy(exec_policy,
                       cnt_iter,
                       cnt_iter + total_row_,
                       thrust::device_pointer_cast(d_random_row),
                       engine);

  cudaStreamSynchronize(sample_stream_);
  shuffle_seed_ = engine();

  if (debug_mode_) {
    int *h_random_row = new int[total_row_ + 10];
    cudaMemcpy(h_random_row,
               d_random_row,
               total_row_ * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < total_row_; xx++) {
      VLOG(2) << "h_random_row[" << xx << "]: " << h_random_row[xx];
    }
    delete[] h_random_row;
    delete[] h_walk;
    delete[] h_sample_keys;
    delete[] h_offset2idx;
    delete[] h_len_per_row;
    delete[] h_prefix_sum;
  }

  if (!sage_mode_) {
    uint64_t h_uniq_node_num = CopyUniqueNodes();
    VLOG(1) << "sample_times:" << sample_times << ", d_walk_size:" << buf_size_
            << ", d_walk_offset:" << i << ", total_rows:" << total_row_
            << ", total_samples:" << total_samples;
  } else {
    VLOG(1) << "sample_times:" << sample_times << ", d_walk_size:" << buf_size_
            << ", d_walk_offset:" << i << ", total_rows:" << total_row_
            << ", total_samples:" << total_samples;
  }
  return total_row_ != 0;
}

int GraphDataGenerator::FillWalkBufMultiPath() {
  platform::CUDADeviceGuard guard(gpuid_);
  size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
  ////////
  uint64_t *h_walk;
  uint64_t *h_sample_keys;
  int *h_offset2idx;
  int *h_len_per_row;
  uint64_t *h_prefix_sum;
  if (debug_mode_) {
    h_walk = new uint64_t[buf_size_];
    h_sample_keys = new uint64_t[once_max_sample_keynum];
    h_offset2idx = new int[once_max_sample_keynum];
    h_len_per_row = new int[once_max_sample_keynum];
    h_prefix_sum = new uint64_t[once_max_sample_keynum + 1];
  }
  ///////
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_->ptr());
  uint8_t *walk_ntype = NULL;
  if (excluded_train_pair_len_ > 0) {
    walk_ntype = reinterpret_cast<uint8_t *>(d_walk_ntype_->ptr());
  }
  int *len_per_row = reinterpret_cast<int *>(d_len_per_row_->ptr());
  uint64_t *d_sample_keys = reinterpret_cast<uint64_t *>(d_sample_keys_->ptr());
  cudaMemsetAsync(walk, 0, buf_size_ * sizeof(uint64_t), sample_stream_);
  int sample_times = 0;
  int i = 0;
  total_row_ = 0;

  // 获取全局采样状态
  auto &first_node_type = gpu_graph_ptr->first_node_type_;
  auto &cur_metapath = gpu_graph_ptr->cur_metapath_;
  auto &meta_path = gpu_graph_ptr->meta_path_;
  auto &path = gpu_graph_ptr->cur_parse_metapath_;
  auto &cur_metapath_start = gpu_graph_ptr->cur_metapath_start_[gpuid_];
  auto &finish_node_type = gpu_graph_ptr->finish_node_type_[gpuid_];
  auto &type_to_index = gpu_graph_ptr->get_graph_type_to_index();
  size_t node_type_len = first_node_type.size();
  std::string first_node =
      paddle::string::split_string<std::string>(cur_metapath, "2")[0];
  auto it = gpu_graph_ptr->node_to_id.find(first_node);
  auto node_type = it->second;

  int remain_size =
      buf_size_ - walk_degree_ * once_sample_startid_len_ * walk_len_;
  int total_samples = 0;

  while (i <= remain_size) {
    size_t start = cur_metapath_start;
    size_t device_key_size = h_train_metapath_keys_len_;
    VLOG(2) << "type: " << node_type << " size: " << device_key_size
            << " start: " << start;
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_train_metapath_keys_->ptr());
    int tmp_len = start + once_sample_startid_len_ > device_key_size
                      ? device_key_size - start
                      : once_sample_startid_len_;
    bool update = true;
    if (tmp_len == 0) {
      break;
    }

    VLOG(2) << "gpuid = " << gpuid_ << " path[0] = " << path[0];
    uint64_t *cur_walk = walk + i;
    uint8_t *cur_walk_ntype = NULL;
    if (excluded_train_pair_len_ > 0) {
      cur_walk_ntype = walk_ntype + i;
    }

    NeighborSampleQuery q;
    q.initialize(gpuid_,
                 path[0],
                 (uint64_t)(d_type_keys + start),
                 walk_degree_,
                 tmp_len);
    auto sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);

    int step = 1;
    VLOG(2) << "sample edge type: " << path[0] << " step: " << 1;
    jump_rows_ = sample_res.total_sample_size;
    total_samples += sample_res.total_sample_size;
    VLOG(2) << "i = " << i << " start = " << start << " tmp_len = " << tmp_len
            << "jump row: " << jump_rows_;
    if (jump_rows_ == 0) {
      cur_metapath_start = tmp_len + start;
      continue;
    }

    if (!sage_mode_) {
      if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
        if (InsertTable(d_type_keys + start, tmp_len, d_uniq_node_num_) != 0) {
          VLOG(2) << "in step 0, insert key stage, table is full";
          update = false;
          break;
        }
        if (InsertTable(sample_res.actual_val,
                        sample_res.total_sample_size,
                        d_uniq_node_num_) != 0) {
          VLOG(2) << "in step 0, insert sample res stage, table is full";
          update = false;
          break;
        }
      }
    }

    FillOneStep(d_type_keys + start,
                path[0],
                cur_walk,
                cur_walk_ntype,
                tmp_len,
                sample_res,
                walk_degree_,
                step,
                len_per_row);
    /////////
    if (debug_mode_) {
      cudaMemcpy(
          h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < buf_size_; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }

    VLOG(2) << "sample, step=" << step << " sample_keys=" << tmp_len
            << " sample_res_len=" << sample_res.total_sample_size;

    /////////
    step++;
    size_t path_len = path.size();
    for (; step < walk_len_; step++) {
      if (sample_res.total_sample_size == 0) {
        VLOG(2) << "sample finish, step=" << step;
        break;
      }
      auto sample_key_mem = sample_res.actual_val_mem;
      uint64_t *sample_keys_ptr =
          reinterpret_cast<uint64_t *>(sample_key_mem->ptr());
      int edge_type_id = path[(step - 1) % path_len];
      VLOG(2) << "sample edge type: " << edge_type_id << " step: " << step;
      q.initialize(gpuid_,
                   edge_type_id,
                   (uint64_t)sample_keys_ptr,
                   1,
                   sample_res.total_sample_size);
      int sample_key_len = sample_res.total_sample_size;
      sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);
      total_samples += sample_res.total_sample_size;
      if (!sage_mode_) {
        if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
          if (InsertTable(sample_res.actual_val,
                          sample_res.total_sample_size,
                          d_uniq_node_num_) != 0) {
            VLOG(2) << "in step: " << step << ", table is full";
            update = false;
            break;
          }
        }
      }
      FillOneStep(d_type_keys + start,
                  edge_type_id,
                  cur_walk,
                  cur_walk_ntype,
                  sample_key_len,
                  sample_res,
                  1,
                  step,
                  len_per_row);
      if (debug_mode_) {
        cudaMemcpy(
            h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        for (int xx = 0; xx < buf_size_; xx++) {
          VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
        }
      }

      VLOG(2) << "sample, step=" << step << " sample_keys=" << sample_key_len
              << " sample_res_len=" << sample_res.total_sample_size;
    }
    // 此时更新全局采样状态
    if (update == true) {
      cur_metapath_start = tmp_len + start;
      i += jump_rows_ * walk_len_;
      total_row_ += jump_rows_;
      sample_times++;
    } else {
      VLOG(2) << "table is full, not update stat!";
      break;
    }
  }
  platform::CUDADeviceGuard guard2(gpuid_);
  buf_state_.Reset(total_row_);
  int *d_random_row = reinterpret_cast<int *>(d_random_row_->ptr());

  thrust::random::default_random_engine engine(shuffle_seed_);
  const auto &exec_policy = thrust::cuda::par.on(sample_stream_);
  thrust::counting_iterator<int> cnt_iter(0);
  thrust::shuffle_copy(exec_policy,
                       cnt_iter,
                       cnt_iter + total_row_,
                       thrust::device_pointer_cast(d_random_row),
                       engine);

  cudaStreamSynchronize(sample_stream_);
  shuffle_seed_ = engine();

  if (debug_mode_) {
    int *h_random_row = new int[total_row_ + 10];
    cudaMemcpy(h_random_row,
               d_random_row,
               total_row_ * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < total_row_; xx++) {
      VLOG(2) << "h_random_row[" << xx << "]: " << h_random_row[xx];
    }
    delete[] h_random_row;
    delete[] h_walk;
    delete[] h_sample_keys;
    delete[] h_offset2idx;
    delete[] h_len_per_row;
    delete[] h_prefix_sum;
  }

  if (!sage_mode_) {
    uint64_t h_uniq_node_num = CopyUniqueNodes();
    VLOG(1) << "sample_times:" << sample_times << ", d_walk_size:" << buf_size_
            << ", d_walk_offset:" << i << ", total_rows:" << total_row_
            << ", h_uniq_node_num:" << h_uniq_node_num
            << ", total_samples:" << total_samples;
  } else {
    VLOG(1) << "sample_times:" << sample_times << ", d_walk_size:" << buf_size_
            << ", d_walk_offset:" << i << ", total_rows:" << total_row_
            << ", total_samples:" << total_samples;
  }

  return total_row_ != 0;
}

void GraphDataGenerator::SetFeedVec(std::vector<phi::DenseTensor *> feed_vec) {
  feed_vec_ = feed_vec;
}

void GraphDataGenerator::AllocResource(
    int thread_id, std::vector<phi::DenseTensor *> feed_vec) {
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  gpuid_ = gpu_graph_ptr->device_id_mapping[thread_id];
  thread_id_ = thread_id;
  place_ = platform::CUDAPlace(gpuid_);
  debug_gpu_memory_info(gpuid_, "AllocResource start");

  platform::CUDADeviceGuard guard(gpuid_);
  if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
    if (gpu_graph_training_) {
      table_ = new HashTable<uint64_t, uint64_t>(
          train_table_cap_ / FLAGS_gpugraph_hbm_table_load_factor);
    } else {
      table_ = new HashTable<uint64_t, uint64_t>(
          infer_table_cap_ / FLAGS_gpugraph_hbm_table_load_factor);
    }
  }
  VLOG(1) << "AllocResource gpuid " << gpuid_
          << " feed_vec.size: " << feed_vec.size()
          << " table cap: " << train_table_cap_;
  sample_stream_ = gpu_graph_ptr->get_local_stream(gpuid_);
  train_stream_ = dynamic_cast<phi::GPUContext *>(
                      platform::DeviceContextPool::Instance().Get(place_))
                      ->stream();
  // feed_vec_ = feed_vec;
  if (!sage_mode_) {
    slot_num_ = (feed_vec.size() - 3) / 2;
  } else {
    slot_num_ = (feed_vec.size() - 4 - samples_.size() * 5) / 2;
  }

  // infer_node_type_start_ = std::vector<int>(h_device_keys_.size(), 0);
  // for (size_t i = 0; i < h_device_keys_.size(); i++) {
  //   for (size_t j = 0; j < h_device_keys_[i]->size(); j++) {
  //     VLOG(3) << "h_device_keys_[" << i << "][" << j
  //             << "] = " << (*(h_device_keys_[i]))[j];
  //   }
  //   auto buf = memory::AllocShared(
  //       place_, h_device_keys_[i]->size() * sizeof(uint64_t));
  //   d_device_keys_.push_back(buf);
  //   CUDA_CHECK(cudaMemcpyAsync(buf->ptr(),
  //                              h_device_keys_[i]->data(),
  //                              h_device_keys_[i]->size() * sizeof(uint64_t),
  //                              cudaMemcpyHostToDevice,
  //                              stream_));
  // }
  if (gpu_graph_training_ && FLAGS_graph_metapath_split_opt) {
    d_train_metapath_keys_ =
        gpu_graph_ptr->d_graph_train_total_keys_[thread_id];
    h_train_metapath_keys_len_ =
        gpu_graph_ptr->h_graph_train_keys_len_[thread_id];
    VLOG(2) << "h train metapaths key len: " << h_train_metapath_keys_len_;
  } else {
    auto &d_graph_all_type_keys = gpu_graph_ptr->d_graph_all_type_total_keys_;
    auto &h_graph_all_type_keys_len = gpu_graph_ptr->h_graph_all_type_keys_len_;

    for (size_t i = 0; i < d_graph_all_type_keys.size(); i++) {
      d_device_keys_.push_back(d_graph_all_type_keys[i][thread_id]);
      h_device_keys_len_.push_back(h_graph_all_type_keys_len[i][thread_id]);
    }
    VLOG(2) << "h_device_keys size: " << h_device_keys_len_.size();
  }

  size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
  d_prefix_sum_ = memory::AllocShared(
      place_,
      (once_max_sample_keynum + 1) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int *d_prefix_sum_ptr = reinterpret_cast<int *>(d_prefix_sum_->ptr());
  cudaMemsetAsync(d_prefix_sum_ptr,
                  0,
                  (once_max_sample_keynum + 1) * sizeof(int),
                  sample_stream_);
  cursor_ = 0;
  jump_rows_ = 0;
  d_uniq_node_num_ = memory::AllocShared(
      place_,
      sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  cudaMemsetAsync(d_uniq_node_num_->ptr(), 0, sizeof(uint64_t), sample_stream_);

  d_walk_ = memory::AllocShared(
      place_,
      buf_size_ * sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  cudaMemsetAsync(
      d_walk_->ptr(), 0, buf_size_ * sizeof(uint64_t), sample_stream_);

  excluded_train_pair_len_ = gpu_graph_ptr->excluded_train_pair_.size();
  if (excluded_train_pair_len_ > 0) {
    d_excluded_train_pair_ = memory::AllocShared(
        place_,
        excluded_train_pair_len_ * sizeof(uint8_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    CUDA_CHECK(cudaMemcpyAsync(d_excluded_train_pair_->ptr(),
                               gpu_graph_ptr->excluded_train_pair_.data(),
                               excluded_train_pair_len_ * sizeof(uint8_t),
                               cudaMemcpyHostToDevice,
                               sample_stream_));

    d_walk_ntype_ = memory::AllocShared(
        place_,
        buf_size_ * sizeof(uint8_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    cudaMemsetAsync(
        d_walk_ntype_->ptr(), 0, buf_size_ * sizeof(uint8_t), sample_stream_);
  }

  d_sample_keys_ = memory::AllocShared(
      place_,
      once_max_sample_keynum * sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));

  d_sampleidx2rows_.push_back(memory::AllocShared(
      place_,
      once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_))));
  d_sampleidx2rows_.push_back(memory::AllocShared(
      place_,
      once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_))));
  cur_sampleidx2row_ = 0;

  d_len_per_row_ = memory::AllocShared(
      place_,
      once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  for (int i = -window_; i < 0; i++) {
    window_step_.push_back(i);
  }
  for (int i = 0; i < window_; i++) {
    window_step_.push_back(i + 1);
  }
  buf_state_.Init(batch_size_, walk_len_, &window_step_);
  d_random_row_ = memory::AllocShared(
      place_,
      (once_sample_startid_len_ * walk_degree_ * repeat_time_) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  shuffle_seed_ = 0;

  ins_buf_pair_len_ = 0;
  if (!sage_mode_) {
    d_ins_buf_ =
        memory::AllocShared(place_, (batch_size_ * 2 * 2) * sizeof(uint64_t));
    d_pair_num_ = memory::AllocShared(place_, sizeof(int));
  } else {
    d_ins_buf_ = memory::AllocShared(
        place_,
        (batch_size_ * 2 * 2) * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    d_pair_num_ = memory::AllocShared(
        place_,
        sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  }

  d_slot_tensor_ptr_ =
      memory::AllocShared(place_, slot_num_ * sizeof(uint64_t *));
  d_slot_lod_tensor_ptr_ =
      memory::AllocShared(place_, slot_num_ * sizeof(uint64_t *));

  if (sage_mode_) {
    reindex_table_size_ = batch_size_ * 2;
    // get hashtable size
    for (int i = 0; i < samples_.size(); i++) {
      reindex_table_size_ *= (samples_[i] * edge_to_id_len_ + 1);
    }
    int64_t next_pow2 =
        1 << static_cast<size_t>(1 + std::log2(reindex_table_size_ >> 1));
    reindex_table_size_ = next_pow2 << 1;

    d_reindex_table_key_ = memory::AllocShared(
        place_,
        reindex_table_size_ * sizeof(int64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    d_reindex_table_value_ = memory::AllocShared(
        place_,
        reindex_table_size_ * sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    d_reindex_table_index_ = memory::AllocShared(
        place_,
        reindex_table_size_ * sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    edge_type_graph_ =
        gpu_graph_ptr->get_edge_type_graph(gpuid_, edge_to_id_len_);

    d_sorted_keys_ = memory::AllocShared(
        place_,
        (batch_size_ * 2 * 2) * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    d_sorted_idx_ = memory::AllocShared(
        place_,
        (batch_size_ * 2 * 2) * sizeof(uint32_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    d_offset_ = memory::AllocShared(
        place_,
        (batch_size_ * 2 * 2) * sizeof(uint32_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    d_merged_cnts_ = memory::AllocShared(
        place_,
        (batch_size_ * 2 * 2) * sizeof(uint32_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  }

  // parse infer_node_type
  auto &type_to_index = gpu_graph_ptr->get_graph_type_to_index();
  if (!gpu_graph_training_) {
    auto node_types =
        paddle::string::split_string<std::string>(infer_node_type_, ";");
    auto node_to_id = gpu_graph_ptr->node_to_id;
    for (auto &type : node_types) {
      auto iter = node_to_id.find(type);
      PADDLE_ENFORCE_NE(
          iter,
          node_to_id.end(),
          platform::errors::NotFound("(%s) is not found in node_to_id.", type));
      int node_type = iter->second;
      int type_index = type_to_index[node_type];
      VLOG(2) << "add node[" << type
              << "] into infer_node_type, type_index(cursor)[" << type_index
              << "]";
      infer_node_type_index_set_.insert(type_index);
    }
    VLOG(2) << "infer_node_type_index_set_num: "
            << infer_node_type_index_set_.size();
  }

  cudaStreamSynchronize(sample_stream_);

  debug_gpu_memory_info(gpuid_, "AllocResource end");
}

void GraphDataGenerator::AllocTrainResource(int thread_id) {
  if (slot_num_ > 0) {
    platform::CUDADeviceGuard guard(gpuid_);
    if (!sage_mode_) {
      d_feature_size_list_buf_ =
          memory::AllocShared(place_, (batch_size_ * 2) * sizeof(uint32_t));
      d_feature_size_prefixsum_buf_ =
          memory::AllocShared(place_, (batch_size_ * 2 + 1) * sizeof(uint32_t));
    } else {
      d_feature_size_list_buf_ = NULL;
      d_feature_size_prefixsum_buf_ = NULL;
    }
  }
}

void GraphDataGenerator::SetConfig(
    const paddle::framework::DataFeedDesc &data_feed_desc) {
  auto graph_config = data_feed_desc.graph_config();
  walk_degree_ = graph_config.walk_degree();
  walk_len_ = graph_config.walk_len();
  window_ = graph_config.window();
  once_sample_startid_len_ = graph_config.once_sample_startid_len();
  debug_mode_ = graph_config.debug_mode();
  gpu_graph_training_ = graph_config.gpu_graph_training();
  if (debug_mode_ || !gpu_graph_training_) {
    batch_size_ = graph_config.batch_size();
  } else {
    batch_size_ = once_sample_startid_len_;
  }
  repeat_time_ = graph_config.sample_times_one_chunk();
  buf_size_ =
      once_sample_startid_len_ * walk_len_ * walk_degree_ * repeat_time_;
  train_table_cap_ = graph_config.train_table_cap();
  infer_table_cap_ = graph_config.infer_table_cap();
  get_degree_ = graph_config.get_degree();
  epoch_finish_ = false;
  VLOG(1) << "Confirm GraphConfig, walk_degree : " << walk_degree_
          << ", walk_len : " << walk_len_ << ", window : " << window_
          << ", once_sample_startid_len : " << once_sample_startid_len_
          << ", sample_times_one_chunk : " << repeat_time_
          << ", batch_size: " << batch_size_
          << ", train_table_cap: " << train_table_cap_
          << ", infer_table_cap: " << infer_table_cap_;
  std::string first_node_type = graph_config.first_node_type();
  std::string meta_path = graph_config.meta_path();
  sage_mode_ = graph_config.sage_mode();
  std::string str_samples = graph_config.samples();
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  debug_gpu_memory_info("init_conf start");
  gpu_graph_ptr->init_conf(
      first_node_type, meta_path, graph_config.excluded_train_pair());
  debug_gpu_memory_info("init_conf end");

  auto edge_to_id = gpu_graph_ptr->edge_to_id;
  edge_to_id_len_ = edge_to_id.size();
  sage_batch_count_ = 0;
  auto samples = paddle::string::split_string<std::string>(str_samples, ";");
  for (size_t i = 0; i < samples.size(); i++) {
    int sample_size = std::stoi(samples[i]);
    samples_.emplace_back(sample_size);
  }
  copy_unique_len_ = 0;

  if (!gpu_graph_training_) {
    infer_node_type_ = graph_config.infer_node_type();
  }
}
#endif

void GraphDataGenerator::DumpWalkPath(std::string dump_path, size_t dump_rate) {
#ifdef _LINUX
  PADDLE_ENFORCE_LT(
      dump_rate,
      10000000,
      platform::errors::InvalidArgument(
          "dump_rate can't be large than 10000000. Please check the dump "
          "rate[1, 10000000]"));
  PADDLE_ENFORCE_GT(dump_rate,
                    1,
                    platform::errors::InvalidArgument(
                        "dump_rate can't be less than 1. Please check "
                        "the dump rate[1, 10000000]"));
  int err_no = 0;
  std::shared_ptr<FILE> fp = fs_open_append_write(dump_path, &err_no, "");
  uint64_t *h_walk = new uint64_t[buf_size_];
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_->ptr());
  cudaMemcpy(
      h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  VLOG(1) << "DumpWalkPath all buf_size_:" << buf_size_;
  std::string ss = "";
  size_t write_count = 0;
  for (int xx = 0; xx < buf_size_ / dump_rate; xx += walk_len_) {
    ss = "";
    for (int yy = 0; yy < walk_len_; yy++) {
      ss += std::to_string(h_walk[xx + yy]) + "-";
    }
    write_count = fwrite_unlocked(ss.data(), 1, ss.length(), fp.get());
    if (write_count != ss.length()) {
      VLOG(1) << "dump walk path" << ss << " failed";
    }
    write_count = fwrite_unlocked("\n", 1, 1, fp.get());
  }
#endif
}

}  // namespace framework
}  // namespace paddle
#endif
