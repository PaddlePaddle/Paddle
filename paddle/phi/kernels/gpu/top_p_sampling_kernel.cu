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

#include "paddle/phi/kernels/top_p_sampling_kernel.h"

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "cub/cub.cuh"
#include "paddle/fluid/operators/kernel_primitives/functor_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/top_k_function_cuda.h"

// #define DEBUG_TOPP

namespace phi {

template <typename T>
struct DataTypeTraits {
  using DataType = T;
};

template <>
struct DataTypeTraits<phi::dtype::float16> {
  using DataType = half;
};

// template <>
// struct DataTypeTraits<phi::dtype::bfloat16> {
//   using DataType = __nv_bfloat16;
// };

#define FINAL_MASK 0xFFFFFFFF

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_BLOCK_DIM(...)                 \
  FIXED_BLOCK_DIM_BASE(1024, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(512, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);   \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)

namespace ops = paddle::operators;

struct SegmentOffsetIter {
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  __host__ __device__ __forceinline__ int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (static_cast<float>(v) < static_cast<float>(value));
  }

  __device__ __forceinline__ bool operator>(const T value) const {
    return (static_cast<float>(v) > static_cast<float>(value));
  }
  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (static_cast<float>(v) < static_cast<float>(in.v)) ||
           ((static_cast<float>(v) == static_cast<float>(in.v)) &&
            (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (static_cast<float>(v) > static_cast<float>(in.v)) ||
           ((static_cast<float>(v) == static_cast<float>(in.v)) &&
            (id < in.id));
  }

  T v;
  int id;
};

inline int div_up(int a, int n) { return (a + n - 1) / n; }

__global__ void setup_kernel(curandState_t* state,
                             const uint64_t seed,
                             const int bs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < bs; i += gridDim.x * blockDim.x) {
    curand_init(seed + i, 0, 0, &state[i]);
  }
}

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[],
                                      const Pair<T>& p,
                                      int beam_size) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(
    Pair<T> topk[], const T* src, int idx, int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[],
                                        const T* src,
                                        int idx,
                                        int dim,
                                        const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[],
                                              int* beam,
                                              int beam_size,
                                              const T* src,
                                              bool* firstStep,
                                              bool* is_empty,
                                              Pair<T>* max,
                                              int dim,
                                              const int tid) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(std::numeric_limits<T>::min(), -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(
            topk + MaxLength - *beam, src, tid, dim, *max, length);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).id == -1) *is_empty = true;
    *beam = 0;
  }
}

template <typename T>
__forceinline__ __device__ Pair<T> WarpReduce(Pair<T> input) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    T tmp_val =
        phi::backends::gpu::CudaShuffleDownSync(FINAL_MASK, input.v, offset);
    int tmp_id =
        phi::backends::gpu::CudaShuffleDownSync(FINAL_MASK, input.id, offset);
    if (static_cast<float>(input.v) < static_cast<float>(tmp_val)) {
      input.v = tmp_val;
      input.id = tmp_id;
    }
  }
  return input;
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void BlockReduce(Pair<T> shared_max[],
                                            Pair<T> topk[],
                                            Pair<T> beam_max[],
                                            int* beam,
                                            int* k,
                                            int* count,
                                            const int tid,
                                            const int wid,
                                            const int lane) {
  while (true) {
    __syncthreads();
    Pair<T> input_now = topk[0];
    input_now = WarpReduce(input_now);

    if (lane == 0) {
      shared_max[wid] = input_now;
    }
    __syncthreads();
    input_now = (tid < BlockSize / 32)
                    ? shared_max[lane]
                    : Pair<T>(std::numeric_limits<T>::min(), -1);
    if (wid == 0) {
      input_now = WarpReduce(input_now);
      if (lane == 0) shared_max[0] = input_now;
    }
    __syncthreads();
    if (tid == 0) {
      beam_max[*count] = shared_max[0];
      (*count)++;
    }
    int tid_max = shared_max[0].id % BlockSize;
    if (tid == tid_max) {
      (*beam)++;
    }
    if (--(*k) == 0) break;
    __syncthreads();

    if (tid == tid_max) {
      if (*beam < MaxLength) {
        topk[0] = topk[*beam];
      }
    }

    if (MaxLength < 5) {
      if (*beam >= MaxLength) break;
    } else {
      unsigned mask = 0u;
      mask = __ballot_sync(FINAL_MASK, true);
      if (tid_max / 32 == wid) {
        if (__shfl_down_sync(FINAL_MASK, *beam, tid_max % 32, 32) == MaxLength)
          break;
      }
    }
  }
}

template <typename T, int MaxLength, int TopPBeamTopK, int BlockSize>
__global__ void KeMatrixTopPBeamTopK(const T* src,
                                     T* top_ps,
                                     int64_t* out_id,  // topk id
                                     T* out_val,       // topk val
                                     int vocab_size,
                                     curandState_t* state,
                                     int* count_iter,
                                     int* count_iter_begin) {
  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lane = tid % 32;
  const int bid = blockIdx.x;

  int top_num = TopPBeamTopK;
  float top_p_num = static_cast<float>(top_ps[bid]);

  __shared__ Pair<T> shared_max[BlockSize / 32];
  __shared__ Pair<T> beam_max[TopPBeamTopK];

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;
  __shared__ int count;

  if (tid == 0) {
    count = 0;
  }

  for (int j = 0; j < MaxLength; j++) {
    topk[j].set(std::numeric_limits<T>::min(), -1);
  }

  while (top_num) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk,
                                           &beam,
                                           TopPBeamTopK,
                                           src + bid * vocab_size,
                                           &firststep,
                                           &is_empty,
                                           &max,
                                           vocab_size,
                                           tid);
    BlockReduce<T, MaxLength, BlockSize>(
        shared_max, topk, beam_max, &beam, &top_num, &count, tid, wid, lane);
  }
  if (tid == 0) {
    count_iter_begin[bid] = count_iter[bid];
    float rand_top_p = curand_uniform(state + bid) * top_p_num;
    top_ps[bid] = (T)rand_top_p;
    float sum_prob = 0.0f;
    for (int i = 0; i < TopPBeamTopK; i++) {
      sum_prob += static_cast<float>(beam_max[i].v);
#ifdef DEBUG_TOPP
      printf("bi: %d, top_p: %f, rand_top_p: %f, sum_prob: %f\n",
             bid,
             top_p_num,
             rand_top_p,
             sum_prob);
#endif
      if (sum_prob >= rand_top_p) {
        count_iter_begin[bid] += 1;
        out_id[bid] = (int64_t)beam_max[i].id;
        out_val[bid] = beam_max[i].v;
#ifdef DEBUG_TOPP
        printf(
            "bi: %d, early stop id: %d\n", bid, static_cast<int>(out_id[bid]));
#endif
        break;
      }
    }
  }
}

__global__ void SetCountIter(int* count_iter, int num) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  for (int i = idx; i < num; i += gridDim.x * blockDim.x) {
    count_iter[i] = i;
  }
}

template <typename T>
__global__ void FillIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (T j = row_id; j < num_rows; j += gridDim.x) {
    for (T i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

struct BlockPrefixCallbackOp {
  float running_total;

  __device__ BlockPrefixCallbackOp(float running_total)
      : running_total(running_total) {}

  __device__ float operator()(float block_aggregate) {
    float old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename T, int BLOCK_SIZE>
__global__ void topp_sampling(T* sorted_probs,
                              int64_t* sorted_id,
                              T* out_val,
                              int64_t* out_id,
                              const T* top_ps,
                              int p_num,
                              int vocab_size,
                              int* count_iter,
                              int* count_iter_begin) {
  __shared__ int stop_shared;
  __shared__ float rand_p;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  const int lane_id = tid % 32;
  const int warp_id = tid / 32;
  const float p_t = static_cast<float>(top_ps[bid]);
  if (tid == 0) {
    stop_shared = 0;
    rand_p = p_t;
#ifdef DEBUG_TOPP
    printf("bi: %d, p: %f\n", bid, rand_p);
#endif
  }
  if (count_iter_begin[bid] == count_iter[bid + 1]) {
    // topk
    return;
  }

  typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ uint32_t selected_shared[NUM_WARPS];

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  if (lane_id == 0) {
    selected_shared[warp_id] = 0;
  }
  __syncthreads();

  int offset = bid * vocab_size;
#ifdef DEBUG_TOPP
  if (tid == 0) {
    printf(
        "first_elem1_1: %f, first_elem1_2: %f, first_id1_1: %d, first_id1_2: "
        "%d\n",
        static_cast<float>(sorted_probs[offset]),
        static_cast<float>(sorted_probs[offset + 1]),
        static_cast<int>(sorted_id[offset]),
        static_cast<int>(sorted_id[offset + 1]);
  }
#endif
  int end = ((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int i_activate = 0;
  float thread_offset = 0;
  for (int i = tid; i < end; i += BLOCK_SIZE) {
    float thread_count =
        (i < vocab_size) ? static_cast<float>(sorted_probs[offset + i]) : 0.f;
    BlockScan(temp_storage)
        .InclusiveSum(thread_count, thread_offset, prefix_op);

    uint32_t activate_mask = __ballot_sync(FINAL_MASK, rand_p <= thread_offset);

    i_activate = i;
    if (activate_mask != 0) {
      if (lane_id == 0) {
        atomicAdd(&stop_shared, 1);
        selected_shared[warp_id] = activate_mask;
      }
    }
    __syncthreads();
    if (stop_shared > 0) {
      break;
    }
  }
  __syncthreads();
  if (stop_shared == 0) {
    if (tid == 0) {
      out_id[bid] = sorted_id[offset + vocab_size - 1];
      out_val[bid] = sorted_probs[offset + vocab_size - 1];
#ifdef DEBUG_TOPP
      printf("stop_shared: %d, out_id: %d, out_val: %f\n",
             static_cast<int>(stop_shared),
             static_cast<int>(out_id[bid]),
             static_cast<float>(out_val[bid]);
#endif
    }
    return;
  }

#ifdef DEBUG_TOPP
  if (tid == 0) {
    printf(
        "first_elem2_1: %f, first_elem2_2: %f, first_id2_1: %d, first_id2_2: "
        "%d\n",
        static_cast<float>(sorted_probs[offset]),
        static_cast<float>(sorted_probs[offset + 1]),
        static_cast<int>(sorted_id[offset]),
        static_cast<int>(sorted_id[offset + 1]);
  }
#endif
  bool skip = (selected_shared[warp_id] > 0) ? false : true;
  for (int i = 0; i < warp_id; i++) {
    if (selected_shared[i] != 0) {
      skip = true;
    }
  }
  if (!skip) {
    int active_lane_id =
        WARP_SIZE - __popc(selected_shared[warp_id]);  // first not 0
    if (lane_id == active_lane_id) {
#ifdef DEBUG_TOPP
      printf(
          "active_lane_id: %d, i_activate: %d.\n", active_lane_id, i_activate);
      for (int i = 0; i < active_lane_id; i++) {
        printf("p %d, value: %f\n",
               i,
               static_cast<float>(sorted_probs[offset + i]));
      }
#endif
      out_id[bid] = sorted_id[offset + i_activate];
      out_val[bid] = sorted_probs[offset + i_activate];
    }
  }
}

int GetBlockSize(int vocab_size) {
  if (vocab_size > 512) {
    return 1024;
  } else if (vocab_size > 256) {
    return 512;
  } else if (vocab_size > 128) {
    return 256;
  } else if (vocab_size > 64) {
    return 128;
  } else {
    return 64;
  }
}

__global__ void set_sorted_num(int* need_sorted_num, int bs) {
  *need_sorted_num = bs;
}

template <typename T>
__global__ void print_kernel(T* input, int size) {
  printf("[");
  for (int i = 0; i < size; i++) {
    if (i != size - 1) {
      printf("%f, ", static_cast<float>(input[i]));
    } else {
      printf("%f]\n", static_cast<float>(input[i]));
    }
  }
}

template <typename T, typename Context>
void TopPSamplingKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& ps,
                        int random_seed,
                        DenseTensor* out,
                        DenseTensor* ids) {
  typedef DataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  auto cu_stream = dev_ctx.stream();
  const auto* input = &x;
  // get the input dims
  const auto& in_dims = input->dims();
  int p_num = ps.numel();
  int bs = in_dims[0];
  int vocab_size = in_dims[1];
  T* out_ptr = dev_ctx.template Alloc<T>(out);
  int64_t* ids_ptr = dev_ctx.template Alloc<int64_t>(ids);

  DenseTensor ps_now;
  ps_now.Resize(phi::make_ddim({bs, 1}));
  dev_ctx.template Alloc<T>(&ps_now);
  phi::Copy(dev_ctx, ps, dev_ctx.GetPlace(), false, &ps_now);

  DenseTensor inds_input;
  inds_input.Resize(phi::make_ddim({bs, vocab_size}));
  dev_ctx.template Alloc<int64_t>(&inds_input);

  DenseTensor sorted_out;
  sorted_out.Resize(phi::make_ddim({bs, vocab_size}));
  dev_ctx.template Alloc<T>(&sorted_out);

  DenseTensor sorted_id;
  sorted_id.Resize(phi::make_ddim({bs, vocab_size}));
  dev_ctx.template Alloc<int64_t>(&sorted_id);

  int BlockSize = GetBlockSize(vocab_size);

  switch (BlockSize) {
    FIXED_BLOCK_DIM(FillIndex<int64_t><<<bs, kBlockDim, 0, cu_stream>>>(
        inds_input.data<int64_t>(), bs, vocab_size));
    default:
      PD_THROW("the input data shape has error in the FillIndex kernel.");
  }

  curandState_t* dev_curand_states;
  phi::Allocator::AllocationPtr curand_states_buf{nullptr};
  curand_states_buf = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      bs * sizeof(curandState_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  dev_curand_states =
      reinterpret_cast<curandState_t*>(curand_states_buf->ptr());
  if (random_seed == -1) {
    srand((unsigned int)(time(NULL)));
    setup_kernel<<<1, 256, 0, cu_stream>>>(dev_curand_states, rand(), bs);
  } else {
    setup_kernel<<<1, 256, 0, cu_stream>>>(dev_curand_states, random_seed, bs);
  }

  DenseTensor count_iter;
  count_iter.Resize(phi::make_ddim({bs + 1}));
  dev_ctx.template Alloc<int>(&count_iter);
  DenseTensor count_iter_begin;
  count_iter_begin.Resize(phi::make_ddim({bs}));
  dev_ctx.template Alloc<int>(&count_iter_begin);
  SetCountIter<<<1, 256, 0, cu_stream>>>(count_iter.data<int>(), bs + 1);

  constexpr int TopKMaxLength = 2;
  constexpr int TopPBeamTopK = 10;
  switch (BlockSize) {
    FIXED_BLOCK_DIM(
        KeMatrixTopPBeamTopK<T, TopKMaxLength, TopPBeamTopK, kBlockDim>
        <<<bs, kBlockDim, 0, cu_stream>>>(x.data<T>(),
                                          ps_now.data<T>(),
                                          ids_ptr,
                                          out_ptr,
                                          vocab_size,
                                          dev_curand_states,
                                          count_iter.data<int>(),
                                          count_iter_begin.data<int>()));
    default:
      PD_THROW("the input data shape has error in the topp_beam_topk kernel.");
  }

  size_t temp_storage_bytes = 0;

  cub::TransformInputIterator<int, SegmentOffsetIter, int*>
      segment_offsets_t_begin(count_iter_begin.data<int>(),
                              SegmentOffsetIter(vocab_size));

  cub::TransformInputIterator<int, SegmentOffsetIter, int*>
      segment_offsets_t_end(count_iter.data<int>(),
                            SegmentOffsetIter(vocab_size));

  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      temp_storage_bytes,
      reinterpret_cast<DataType_*>(const_cast<T*>(x.data<T>())),
      reinterpret_cast<DataType_*>(const_cast<T*>(sorted_out.data<T>())),
      inds_input.data<int64_t>(),
      sorted_id.data<int64_t>(),
      vocab_size * bs,
      bs,
      segment_offsets_t_begin,
      segment_offsets_t_end + 1,
      0,
      sizeof(T) * 8,
      cu_stream);

  temp_storage_bytes = div_up(temp_storage_bytes, 256) * 256;
  int64_t temp_size = temp_storage_bytes;
  DenseTensor temp_storage;
  temp_storage.Resize(phi::make_ddim({temp_size}));
  dev_ctx.template Alloc<uint8_t>(&temp_storage);

  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      temp_storage.data<uint8_t>(),
      temp_storage_bytes,
      reinterpret_cast<DataType_*>(const_cast<T*>(x.data<T>())),
      reinterpret_cast<DataType_*>(const_cast<T*>(sorted_out.data<T>())),
      inds_input.data<int64_t>(),
      sorted_id.data<int64_t>(),
      vocab_size * bs,
      bs,
      segment_offsets_t_begin,
      segment_offsets_t_end + 1,
      0,
      sizeof(T) * 8,
      cu_stream);
  switch (BlockSize) {
    FIXED_BLOCK_DIM(
        topp_sampling<T, kBlockDim>
        <<<bs, kBlockDim, 0, cu_stream>>>(sorted_out.data<T>(),
                                          sorted_id.data<int64_t>(),
                                          out_ptr,
                                          ids_ptr,
                                          ps_now.data<T>(),
                                          p_num,
                                          vocab_size,
                                          count_iter.data<int>(),
                                          count_iter_begin.data<int>()));
    default:
      PD_THROW("the input data shape has error in the topp_sampling kernel.");
  }
  return;
}

}  // namespace phi

PD_REGISTER_KERNEL(top_p_sampling,
                   GPU,
                   ALL_LAYOUT,
                   phi::TopPSamplingKernel,
                   float,
                   phi::dtype::float16) {}
