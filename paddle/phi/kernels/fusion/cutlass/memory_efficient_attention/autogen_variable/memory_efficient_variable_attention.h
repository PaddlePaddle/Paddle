
#pragma once

#ifdef PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "cutlass/util/device_memory.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/default_fmha_grouped.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/gemm/gemm_grouped.h"

namespace phi {

using GemmCoord = cutlass::gemm::GemmCoord;

struct Params {
  // meta params
  phi::DataType datatype;

  // [bs, nh, seq_len, dh]
  const void* query_ptr;
  const void* key_ptr;
  const void* value_ptr;

  // and it can be broadcasted in axis0, 1, 2.
  const void* mask_ptr = nullptr;

  const int* seq_lens = nullptr;
  const int* kv_seq_lens = nullptr;

  // Output tensors
  void* output_ptr;  // [num_batches, num_heads, query_seq_len, head_size]
  void* output_accum_ptr =
      nullptr;  // [num_batches, num_heads, query_seq_len, head_size]

  // Scale
  float scale;

  // Dimensions/strides
  int32_t num_batches;
  int32_t num_heads;
  int32_t query_seq_len;
  int32_t key_value_seq_len;
  int32_t head_size;
  int32_t value_head_size;

  int64_t ldq;
  int64_t ldk;
  int64_t ldm;
  int64_t ldv;
  int64_t ldo;

  int64_t ElementQ;
  int64_t ElementK;
  int64_t ElementM;
  int64_t ElementV;
  int64_t ElementO;

  bool causal;
  bool mask_broadcast_row;
};

__global__ static void get_problem_sizes(const int* seq_lens,
                                         const int* kv_seq_lens,
                                         GemmCoord* problem_sizes0,
                                         GemmCoord* problem_sizes1,
                                         const int bs,
                                         const int num_head,
                                         const int head_size,
                                         const int value_head_size) {
  int bi = blockIdx.x;
  int hi = threadIdx.x;
  if (bi < bs && hi < num_head) {
    int id = bi * num_head + hi;
    int m = seq_lens[bi];
    int mkv = kv_seq_lens[bi];
    int k0 = head_size;
    int k1 = value_head_size;
    GemmCoord problem0(m, mkv, k0);
    GemmCoord problem1(m, k1, mkv);
    problem_sizes0[id] = problem0;
    problem_sizes1[id] = problem1;
  }
}

template <typename T>
struct CutlassTrait {
  using Type = T;
};

template <>
struct CutlassTrait<dtype::float16> {
  using Type = cutlass::half_t;
};

template <>
struct CutlassTrait<dtype::bfloat16> {
  using Type = cutlass::bfloat16_t;
};


template <typename T>
struct ToPhiDTypeTrait {
 private:
  using NonConstT = typename std::remove_const<T>::type;
  static constexpr bool kIsFP16 = std::is_same<NonConstT, cutlass::half_t>::value;
  static constexpr bool kIsBF16 = std::is_same<NonConstT, cutlass::bfloat16_t>::value;

 public:
  using Type = typename std::conditional<kIsFP16, dtype::float16,
      typename std::conditional<kIsBF16, dtype::bfloat16, NonConstT>::type>::type;
};

} // namespace phi

#include "./cutlass_forward.h"

#endif
