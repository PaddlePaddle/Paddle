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

#include <vector>

#include "cub/cub.cuh"

#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
<<<<<<< HEAD
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/device_context.h"
=======
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

#include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"
#include "paddle/fluid/operators/fused_token_prune_op.cu.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

<<<<<<< HEAD
#if IS_TRT_VERSION_GE(6000)

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
template <typename T>
__global__ void ElementwiseMask(const T* a,
                                const T* b,
                                T* res,
                                int num_elements) {
<<<<<<< HEAD
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_elements) return;
  const T zero = 0;
  res[tid] = b[tid] >= zero ? a[tid] : zero;
<<<<<<< HEAD
#endif
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
}

template <typename T>
__global__ void FillZero(T* data, int len) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  const T zero = 0;
  data[tid] = zero;
}

__global__ void FillIndex(int32_t* indices, int num_raws, int num_cols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_raws * num_cols) return;

  int col = tid % num_cols;
  int raw = tid / num_cols;

  indices[tid] = col;
}

template <typename T>
__global__ void MaximumFirst(T* mat, int num_raws, int num_cols, T max_value) {
  auto raw = blockIdx.x * blockDim.x + threadIdx.x;
  if (raw >= num_raws) return;
  mat[raw * num_cols] = max_value;
}

__global__ void FillOffsets(int* offsets, int num_raws, int num_cols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid > num_raws) return;

  offsets[tid] = tid * num_cols;
}

template <typename T>
__global__ void Slice(
    const T* src, T* dst, int num_raws, int src_num_cols, int dst_num_cols) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_raws * dst_num_cols) return;
  int raw = tid / dst_num_cols;
  int col = tid % dst_num_cols;
  dst[tid] = src[raw * src_num_cols + col];
}

template <typename T>
__global__ void ReduceSum2(
    const T* src, T* dst, int bsz, int nb_head, int max_seq_len) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_blocks_per_head = ((max_seq_len / blockDim.x) * max_seq_len);
  int batch = bid / (nb_head * num_blocks_per_head);
  int col = bid % max_seq_len;
  int head = (bid / num_blocks_per_head) % nb_head;

  extern __shared__ T res_float[];
  res_float[tid] =
      src[batch * (nb_head * max_seq_len * max_seq_len) +
          head * (max_seq_len * max_seq_len) + col + tid * max_seq_len];
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      res_float[tid] += res_float[tid + offset];
    }
    __syncthreads();
    if (offset % 2 == 1 && tid == offset - 2) {
      res_float[tid] += res_float[tid + 1];
    }
  }

  if (tid == 0) {
    auto* dst_addr = dst + batch * max_seq_len + col;
    atomicAdd(dst_addr, res_float[0]);
  }
}

template <>
__global__ void ReduceSum2<half>(
    const half* src, half* dst, int bsz, int nb_head, int max_seq_len) {
<<<<<<< HEAD
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_blocks_per_head = ((max_seq_len / blockDim.x) * max_seq_len);
  int batch = bid / (nb_head * num_blocks_per_head);
  int col = bid % max_seq_len;
  int head = (bid / num_blocks_per_head) % nb_head;

  extern __shared__ half res_half[];
  res_half[tid] =
      src[batch * (nb_head * max_seq_len * max_seq_len) +
          head * (max_seq_len * max_seq_len) + col + tid * max_seq_len];
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      res_half[tid] += res_half[tid + offset];
    }
    __syncthreads();
    if (offset % 2 == 1 && tid == offset - 2) {
      res_half[tid] += res_half[tid + 1];
    }
    __syncthreads();
  }

  if (tid == 0) {
<<<<<<< HEAD
    platform::fastAtomicAdd<platform::float16>(
=======
    phi::fastAtomicAdd<platform::float16>(
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        reinterpret_cast<platform::float16*>(dst),
        static_cast<size_t>(batch * max_seq_len + col),
        static_cast<size_t>(bsz * max_seq_len),
        static_cast<platform::float16>(res_half[0]));
  }
<<<<<<< HEAD
#endif
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
}

template <typename T>
__global__ void TakeAlongAxis(const T* src,
                              T* dst,
                              int32_t* indices,
                              int num_raws,
                              int src_num_cols,
                              int dst_num_cols,
                              int num_elements) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= num_raws * dst_num_cols) return;

  int raw = tid / dst_num_cols;
  int col = tid % dst_num_cols;
  for (int i = 0; i < num_elements; ++i) {
    dst[tid * num_elements + i] =
        *(src + (raw * src_num_cols + indices[tid]) * num_elements + i);
  }
}

<<<<<<< HEAD
__global__ void pos_id_prune_kernel(const int32_t* src,
                                    int32_t* dst,
                                    int pos_nums,
                                    float scale) {
  dst[0] = 0;
  for (int i = 1; i < pos_nums; i++) {
    dst[i] =
        dst[i - 1] + max(static_cast<int>((src[i] - src[i - 1]) * scale), 2);
=======
__global__ void compute_token_length(const int32_t* src,
                                     int32_t* dst,
                                     float scale) {
  int32_t it = threadIdx.x;
  dst[it] = max(static_cast<int>((src[it + 1] - src[it]) * scale), 1);
}

__global__ void fill_index_padding_score(int32_t* token_index,
                                         const half* scores,
                                         int32_t scores_size,
                                         half* padding_scores) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  token_index[tid] = threadIdx.x;
  if (tid < scores_size) {
    padding_scores[tid] = scores[tid];
  } else {
    padding_scores[tid] = 0;
  }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void general_topk_pair_sort(T* in_keys, int32_t* in_out_values) {
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
      .Load(in_out_values + block_offset, thread_values);
  __syncthreads();

  BlockRadixSort(temp_storage.sort).SortDescending(thread_keys, thread_values);
  __syncthreads();

  BlockStoreValue(temp_storage.storevalue)
      .Store(in_out_values + block_offset, thread_values);
}

__global__ void varlen_prune_token(const half* tokens,
                                   const int32_t* token_pos,
                                   const int32_t* token_index,
                                   half* output) {
  int batch = blockIdx.x;
  int token_it = batch * gridDim.y + blockIdx.y;
  int pre_value_it =
      token_it * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;

  if (token_index[token_it] < token_pos[batch + 1] - token_pos[batch]) {
    output[(token_index[token_it] + token_pos[batch]) * gridDim.z * blockDim.x +
           blockIdx.z * blockDim.x + threadIdx.x] = tokens[pre_value_it];
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }
}

nvinfer1::DimsExprs FusedTokenPrunePluginDynamic::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  auto x_dims = inputs[1], new_mask_dims = inputs[3];
  if (flag_varseqlen_) {
<<<<<<< HEAD
    if (output_index == 0) {
      nvinfer1::DimsExprs ret = x_dims;
      ret.d[1] = new_mask_dims.d[2];
=======
    // max sum of seqlen: ceil(sum / scale) + n -1 >= for(i=0;i<n;i++) {sum +=
    // floor(num(i) / scale)} auto
    // pruned_sum_length=std::ceil(inputs[4].d[0]*new_mask_dims.d[2]/inputs[6].d[1])+
    // inputs[1].d[0] - 1;
    auto pruned_sum_length = expr_builder.operation(
        nvinfer1::DimensionOperation::kSUB,
        *expr_builder.operation(
            nvinfer1::DimensionOperation::kSUM,
            *expr_builder.operation(
                nvinfer1::DimensionOperation::kCEIL_DIV,
                *expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                        *inputs[4].d[0],
                                        *new_mask_dims.d[2]),
                *inputs[6].d[1]),
            *inputs[1].d[0]),
        *expr_builder.constant(1));
    if (output_index == 0) {
      nvinfer1::DimsExprs ret;
      ret.nbDims = 4;
      ret.d[0] = pruned_sum_length;
      ret.d[1] = x_dims.d[2];
      ret.d[2] = expr_builder.constant(1);
      ret.d[3] = expr_builder.constant(1);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
      return ret;
    } else if (output_index == 1) {
      nvinfer1::DimsExprs ret;
      ret.nbDims = 2;
      ret.d[0] = new_mask_dims.d[0];
      ret.d[1] = new_mask_dims.d[2];
      return ret;
    } else if (output_index == 2) {
      // word id
      nvinfer1::DimsExprs ret;
      ret.nbDims = 1;
<<<<<<< HEAD
      // max sum of seqlen: pre_seqlen * new_mask[2] / mask[1] + 2 * batchs
      const auto* two = expr_builder.constant(2);
      ret.d[0] = expr_builder.operation(
          nvinfer1::DimensionOperation::kSUM,
          *expr_builder.operation(
              nvinfer1::DimensionOperation::kFLOOR_DIV,
              *expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                      *inputs[4].d[0],
                                      *new_mask_dims.d[2]),
              *inputs[6].d[1]),
          *expr_builder.operation(
              nvinfer1::DimensionOperation::kPROD, *two, *inputs[6].d[0]));
=======
      ret.d[0] = pruned_sum_length;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
      return ret;
    } else if (output_index == 3) {
      // pos id
      nvinfer1::DimsExprs ret = inputs[5];
      return ret;
    } else if (output_index == 4) {
      // mask id
      nvinfer1::DimsExprs ret;
      ret.nbDims = 2;
      ret.d[0] = inputs[6].d[0];
      ret.d[1] = new_mask_dims.d[2];
      return ret;
    }
  } else {
    if (output_index == 0) {
      nvinfer1::DimsExprs ret = x_dims;
      ret.d[1] = new_mask_dims.d[2];
      return ret;
    } else {
      nvinfer1::DimsExprs ret;
      ret.nbDims = 2;
      ret.d[0] = new_mask_dims.d[0];
      ret.d[1] = new_mask_dims.d[2];
      return ret;
    }
  }
}

bool FusedTokenPrunePluginDynamic::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out,
      platform::errors::InvalidArgument(
          "The input of swish plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos,
      nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos,
                                        nb_inputs + nb_outputs));

  const nvinfer1::PluginTensorDesc& in = in_out[pos];
  if (flag_varseqlen_) {
<<<<<<< HEAD
    if (pos == 0) {
      if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
        return (in.type == nvinfer1::DataType::kFLOAT ||
                in.type == nvinfer1::DataType::kHALF) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
        return (in.type == nvinfer1::DataType::kFLOAT) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
#endif
      } else {
        return (in.type == nvinfer1::DataType::kFLOAT) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
      }
    } else if (pos <= 3 || pos == 7) {
      const nvinfer1::PluginTensorDesc& prev = in_out[0];
      return in.type == prev.type && in.format == prev.format;
    } else if (pos == 6 || pos == 11) {  // mask_id, mask_id_out
      return in.type == nvinfer1::DataType::kFLOAT &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
=======
    if (pos <= 3 || pos == 7) {
      if (with_fp16_) {
        return (in.type == nvinfer1::DataType::kHALF) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
      } else {
        PADDLE_THROW(platform::errors::Fatal(
            "The FusedTokenPrune TRT Plugin's input type "
            "should be half for varseqlen."));
      }
    } else if (pos == 6 || pos == 11) {  // mask_id, mask_id_out
      return (in.type == nvinfer1::DataType::kFLOAT) &&
             (in.format == nvinfer1::TensorFormat::kLINEAR);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    } else {
      return in.type == nvinfer1::DataType::kINT32 &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  } else {
    if (pos == 0) {
      if (with_fp16_) {
<<<<<<< HEAD
#ifdef TRT_PLUGIN_FP16_AVALIABLE
        return (in.type == nvinfer1::DataType::kFLOAT ||
                in.type == nvinfer1::DataType::kHALF) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
#else
        return (in.type == nvinfer1::DataType::kFLOAT) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
#endif
=======
        return (in.type == nvinfer1::DataType::kHALF) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);

>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
      } else {
        return (in.type == nvinfer1::DataType::kFLOAT) &&
               (in.format == nvinfer1::TensorFormat::kLINEAR);
      }
    } else if (pos <= 4) {
      const nvinfer1::PluginTensorDesc& prev = in_out[0];
      return in.type == prev.type && in.format == prev.format;
    } else {
      return in.type == nvinfer1::DataType::kINT32 &&
             in.format == nvinfer1::TensorFormat::kLINEAR;
    }
  }
}

nvinfer1::DataType FusedTokenPrunePluginDynamic::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  if (flag_varseqlen_) {
    if (index == 0) {
<<<<<<< HEAD
      return input_types[1];
    } else if (index == 4) {
      return nvinfer1::DataType::kFLOAT;
=======
      return nvinfer1::DataType::kHALF;
    } else if (index == 4) {  // mask id
      return input_types[6];
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    } else {
      // index = 1,2,3
      return nvinfer1::DataType::kINT32;
    }
  } else {
    if (index == 0) {
      return input_types[1];
    } else {
      // index = 1
      return nvinfer1::DataType::kINT32;
    }
  }
}

size_t FusedTokenPrunePluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nb_outputs) const TRT_NOEXCEPT {
  auto attn_dims = inputs[0].dims;
  auto x_dims = inputs[1].dims;
  auto new_mask_dims = inputs[3].dims;
  auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
       max_seq_len = attn_dims.d[2];

  int slimmed_x_len = new_mask_dims.d[2];
  int total = bsz * nb_head * max_seq_len * max_seq_len;
  size_t size = total * sizeof(float);
  size += bsz * max_seq_len * sizeof(float);
  size += bsz * max_seq_len * sizeof(int32_t);
  size += bsz * max_seq_len * sizeof(float);
  size += bsz * max_seq_len * sizeof(int32_t);
  size += (bsz + 1) * sizeof(int);
  size += bsz * slimmed_x_len * sizeof(int32_t);
  return size;
}

template <typename T>
inline void enqueueImpl(const nvinfer1::PluginTensorDesc* input_desc,
                        const nvinfer1::PluginTensorDesc* output_desc,
                        const void* const* inputs,
                        void* const* outputs,
                        void* workspace_ptr,
                        cudaStream_t stream,
                        int device_id,
                        T max_value,
                        bool keep_first_token_,
                        bool keep_order_) {
  // Dims
  auto attn_dims = input_desc[0].dims;
  auto x_dims = input_desc[1].dims;
  auto new_mask_dims = input_desc[3].dims;

  auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
       max_seq_len = attn_dims.d[2];
  auto c = x_dims.d[2];
  auto slimmed_x_len = new_mask_dims.d[2];

  // Inputs
  const T* attn_data = static_cast<const T*>(inputs[0]);
  const T* x_data = static_cast<const T*>(inputs[1]);
  const T* mask_data = static_cast<const T*>(inputs[2]);

  // Outputs
  T* output_data = static_cast<T*>(outputs[0]);
  int32_t* output_indices_data = static_cast<int32_t*>(outputs[1]);

  int total = bsz * nb_head * max_seq_len * max_seq_len;
  int block = operators::ComputeBlockSize(max_seq_len);
  int grid = operators::CeilDivide(total, block);

  // Workspace for intermediate variable
  char* workspace = static_cast<char*>(workspace_ptr);
  T* attn_tmp_data = reinterpret_cast<T*>(workspace);
  size_t offset = total * sizeof(T);
  T* attn_accu_data = reinterpret_cast<T*>(workspace + offset);
  offset += bsz * max_seq_len * sizeof(T);
  int32_t* attn_accu_indices_data =
      reinterpret_cast<int32_t*>(workspace + offset);
  offset += bsz * max_seq_len * sizeof(int32_t);
  T* sort_attn_accu_data = reinterpret_cast<T*>(workspace + offset);
  offset += bsz * max_seq_len * sizeof(T);
  int32_t* sort_attn_accu_indices_data =
      reinterpret_cast<int32_t*>(workspace + offset);
  offset += bsz * max_seq_len * sizeof(int32_t);
  int* offsets_data = reinterpret_cast<int*>(workspace + offset);
  offset += (bsz + 1) * sizeof(int);
  int32_t* slimmed_sort_attn_accu_indices_data =
      reinterpret_cast<int32_t*>(workspace + offset);

  // 1. Filter attn by mask
  ElementwiseMask<T>
      <<<grid, block, 0, stream>>>(attn_data, mask_data, attn_tmp_data, total);

  total = bsz * max_seq_len;
  block = operators::ComputeBlockSize(max_seq_len);
  grid = operators::CeilDivide(total, block);
  FillZero<T><<<grid, block, 0, stream>>>(attn_accu_data, total);

  // 2. Reduce sum
  total = bsz * nb_head * max_seq_len * max_seq_len;
  int block_tmp = max_seq_len;
  while (block_tmp > 1024)
    block_tmp /= 2;  // if max seq len > 1024, it must be 2^n
  block =
      block_tmp;  // make sure max_seq_len is an integral multiple of block_size
  grid = operators::CeilDivide(total, block);
  ReduceSum2<T><<<grid, block, block * sizeof(T), stream>>>(
      attn_tmp_data, attn_accu_data, bsz, nb_head, max_seq_len);

  // 3. Prepare token indices
  total = bsz * max_seq_len;
  block = operators::ComputeBlockSize(max_seq_len);
  grid = operators::CeilDivide(total, block);

  FillIndex<<<grid, block, 0, stream>>>(
      attn_accu_indices_data, bsz, max_seq_len);

  // 4. Sort token indices by attn
  if (keep_first_token_) {
    MaximumFirst<T>
        <<<bsz, 1, 0, stream>>>(attn_accu_data, bsz, max_seq_len, max_value);
  }
  size_t temp_storage_bytes = -1;
  int num_items = bsz * max_seq_len;
  int num_segments = bsz;
  FillOffsets<<<bsz + 1, 1, 0, stream>>>(offsets_data, bsz, max_seq_len);
  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_storage_bytes,
      attn_accu_data,
      sort_attn_accu_data,
      attn_accu_indices_data,
      sort_attn_accu_indices_data,
      num_items,
      num_segments,
      offsets_data,
      offsets_data + 1,
      0,
      sizeof(T) * 8,
      stream));
  int64_t temp_size = temp_storage_bytes;
<<<<<<< HEAD
  framework::Tensor temp_storage;
=======
  phi::DenseTensor temp_storage;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  auto* temp_storage_data = temp_storage.mutable_data<uint8_t>(
      {temp_size}, platform::CUDAPlace(device_id));

  PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp_storage_data,
      temp_storage_bytes,
      attn_accu_data,
      sort_attn_accu_data,
      attn_accu_indices_data,
      sort_attn_accu_indices_data,
      num_items,
      num_segments,
      offsets_data,
      offsets_data + 1,
      0,
      sizeof(T) * 8,
      stream));
  // 5. Slice
  total = bsz * slimmed_x_len;
  block = operators::ComputeBlockSize(slimmed_x_len);
  grid = operators::CeilDivide(total, block);

  Slice<int32_t>
      <<<grid, block, 0, stream>>>(sort_attn_accu_indices_data,
                                   slimmed_sort_attn_accu_indices_data,
                                   bsz,
                                   max_seq_len,
                                   slimmed_x_len);

  if (keep_order_) {
    // 6. reorder
    num_items = bsz * slimmed_x_len;
    FillOffsets<<<bsz + 1, 1, 0, stream>>>(offsets_data, bsz, slimmed_x_len);
    temp_storage_bytes = -1;
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr,
        temp_storage_bytes,
        slimmed_sort_attn_accu_indices_data,
        output_indices_data,
        num_items,
        num_segments,
        offsets_data,
        offsets_data + 1,
        0,
        sizeof(int32_t) * 8,
        stream));

    temp_size = temp_storage_bytes;
    temp_storage.Resize({temp_size});
    temp_storage_data =
        temp_storage.mutable_data<uint8_t>(platform::CUDAPlace(device_id));
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceSegmentedRadixSort::SortKeys(
        temp_storage_data,
        temp_storage_bytes,
        slimmed_sort_attn_accu_indices_data,
        output_indices_data,
        num_items,
        num_segments,
        offsets_data,
        offsets_data + 1,
        0,
        sizeof(int32_t) * 8,
        stream));

    TakeAlongAxis<T><<<grid, block, 0, stream>>>(x_data,
                                                 output_data,
                                                 output_indices_data,
                                                 bsz,
                                                 max_seq_len,
                                                 slimmed_x_len,
                                                 c);
  } else {
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(output_indices_data,
                                          slimmed_sort_attn_accu_indices_data,
                                          bsz * slimmed_x_len * sizeof(int32_t),
                                          cudaMemcpyDeviceToDevice));
    TakeAlongAxis<T>
        <<<grid, block, 0, stream>>>(x_data,
                                     output_data,
                                     slimmed_sort_attn_accu_indices_data,
                                     bsz,
                                     max_seq_len,
                                     slimmed_x_len,
                                     c);
  }
}

<<<<<<< HEAD
inline void pos_id_prune(const int32_t* input,
                         int32_t* output,
                         int pos_nums,
                         float scale,
                         cudaStream_t stream) {
  pos_id_prune_kernel<<<1, 1, 0, stream>>>(input, output, pos_nums, scale);
}

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
int FusedTokenPrunePluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
<<<<<<< HEAD
  auto input_type = input_desc[0].type;
  auto attn_dims = input_desc[0].dims;
  auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
       max_seq_len = attn_dims.d[2];
  int device_id;
  cudaGetDevice(&device_id);

  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp32";

    float max = std::numeric_limits<float>::max();

    enqueueImpl<float>(input_desc,
                       output_desc,
                       inputs,
                       outputs,
                       workspace,
                       stream,
                       device_id,
                       max,
                       keep_first_token_,
                       keep_order_);

  } else if (input_type == nvinfer1::DataType::kHALF) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
    VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp16";

    half max = 65504.0;

    enqueueImpl<half>(input_desc,
                      output_desc,
                      inputs,
                      outputs,
                      workspace,
                      stream,
                      device_id,
                      max,
                      keep_first_token_,
                      keep_order_);

#else
    PADDLE_THROW(platform::errors::Fatal(
        "The Ernie(Bert) TensorRT Plugin should be "
        "complied with CUDA version >= 10.0 when running with fp16. "
        "Please recomplie it or try to use fp32 by set "
        "config.SetTRTDynamicShapeInfo(min_input_shape, "
        "max_input_shape, opt_input_shape, true"));
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Fatal("The FusedTokenPrune TRT Plugin's input type "
                                "should be float or half."));
  }
  if (flag_varseqlen_) {
    float scale =
        static_cast<float>(input_desc[3].dims.d[2]) / input_desc[6].dims.d[1];
    // outputs[2]=inputs[4]; // word_id
    const int32_t* inputs5 = static_cast<const int32_t*>(inputs[5]);
    int32_t* outputs3 = static_cast<int32_t*>(outputs[3]);
    pos_id_prune(
        inputs5, outputs3, input_desc[5].dims.d[0], scale, stream);  // pos_id
    // outputs[4]=inputs[6]; // new_mask
=======
  if (flag_varseqlen_) {
    if (!(input_desc[0].type == nvinfer1::DataType::kHALF &&
          input_desc[1].type == nvinfer1::DataType::kHALF)) {
      PADDLE_THROW(
          platform::errors::InvalidArgument("Token_prune'type must half"));
    }
    float scale =
        static_cast<float>(input_desc[3].dims.d[2]) / input_desc[6].dims.d[1];
    const int32_t* inputs5 =
        static_cast<const int32_t*>(inputs[5]);             // pre pos id
    int32_t* outputs3 = static_cast<int32_t*>(outputs[3]);  // new pos id
    half* outputs0 = static_cast<half*>(outputs[0]);

    const int32_t B = input_desc[1].dims.d[0];  // batchs
    const int32_t max_sequnce_length =
        input_desc[1].dims.d[1];                     // max sequnce length
    const int32_t length = input_desc[1].dims.d[2];  // vector length
    const half* scores = static_cast<const half*>(inputs[0]);  // reduce sum
    const half* tokens = static_cast<const half*>(inputs[1]);
    const int32_t scores_size = B * max_sequnce_length;
    int32_t padding_token_length;
    if (max_sequnce_length <= 128) {
      padding_token_length = 128;
    } else if (max_sequnce_length <= 256) {
      padding_token_length = 256;
    } else if (max_sequnce_length <= 384) {
      padding_token_length = 384;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Token_prune'token_length must <= 384"));
    }

    // 1. Compute the token length after pruning.
    compute_token_length<<<1, B, 0, stream>>>(
        inputs5, pruned_token_lengths_, scale);

    fill_index_padding_score<<<B, padding_token_length, 0, stream>>>(
        token_index_, scores, scores_size, padding_scores_);

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  pruned_token_lengths_,
                                  outputs3,
                                  B + 1);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  pruned_token_lengths_,
                                  outputs3,
                                  B + 1);

    if (padding_token_length == 128) {
      general_topk_pair_sort<half, 32, 4>
          <<<B, 32, 0, stream>>>(padding_scores_, token_index_);  // 128
    } else if (padding_token_length == 256) {
      general_topk_pair_sort<half, 64, 4>
          <<<B, 64, 0, stream>>>(padding_scores_, token_index_);  // 256
    } else {
      general_topk_pair_sort<half, 96, 4>
          <<<B, 96, 0, stream>>>(padding_scores_, token_index_);  // 384
    }

    int32_t num_threads;
    if (length < 1024) {
      num_threads = length;
    } else {
      if (length % 512 == 0) {
        num_threads = 512;
      } else if (length % 256 == 0) {
        num_threads = 256;
      } else if (length % 128 == 0) {
        num_threads = 128;
      } else if (length % 64 == 0) {
        num_threads = 64;
      } else if (length % 32 == 0) {
        num_threads = 32;
      } else if (length % 16 == 0) {
        num_threads = 16;
      } else if (length % 8 == 0) {
        num_threads = 8;
      } else if (length % 4 == 0) {
        num_threads = 4;
      } else if (length % 2 == 0) {
        num_threads = 2;
      } else {
        num_threads = 1;
      }
    }
    const dim3 num_blocks(
        B,
        max_sequnce_length,
        length / num_threads);  //  batchs, max_sequnce_length, vector_ength/***
    varlen_prune_token<<<num_blocks, num_threads, 0, stream>>>(
        tokens, outputs3, token_index_, outputs0);
  } else {
    auto input_type = input_desc[0].type;
    auto attn_dims = input_desc[0].dims;
    auto bsz = attn_dims.d[0], nb_head = attn_dims.d[1],
         max_seq_len = attn_dims.d[2];
    int device_id;
    cudaGetDevice(&device_id);

    if (input_type == nvinfer1::DataType::kFLOAT) {
      VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp32";

      float max = std::numeric_limits<float>::max();

      enqueueImpl<float>(input_desc,
                         output_desc,
                         inputs,
                         outputs,
                         workspace,
                         stream,
                         device_id,
                         max,
                         keep_first_token_,
                         keep_order_);

    } else if (input_type == nvinfer1::DataType::kHALF) {
      VLOG(1) << "TRT Plugin DataType selected. FusedTokenPrune-->fp16";

      half max = 65504.0;
      enqueueImpl<half>(input_desc,
                        output_desc,
                        inputs,
                        outputs,
                        workspace,
                        stream,
                        device_id,
                        max,
                        keep_first_token_,
                        keep_order_);
    } else {
      PADDLE_THROW(
          platform::errors::Fatal("The FusedTokenPrune TRT Plugin's input type "
                                  "should be float or half."));
    }
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }
  return cudaGetLastError() != cudaSuccess;
}

<<<<<<< HEAD
#endif
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
