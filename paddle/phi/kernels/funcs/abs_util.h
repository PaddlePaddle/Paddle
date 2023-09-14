// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <typeinfo>
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/function_traits.h"

namespace phi {
namespace funcs {

// template <typename T, typename Func>
// __global__ void abs_kernel(const T* x, const int num, Func f, T* out) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < num) {
//     out[idx] = f(x[idx]);
//   }
// }

struct TensorContainer {
  explicit TensorContainer(const DenseTensor* x, DenseTensor* out)
      : x_(x), out_(out) {}

  const DenseTensor* x_;
  DenseTensor* out_;
};

template <typename T, int kVecSize>
struct alignas(sizeof(T) * kVecSize) AlignedVector {
  T val_[kVecSize];
};

template <typename T>
int GetVectorizedSize(const void* pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  constexpr int vec2 = std::alignment_of<AlignedVector<T, 2>>::value;  // NOLINT
  /*
    * Currently, decide to deal with no more than 4 data once while adopting
    * vectorization load/store, if performance test shows that dealing with
    * 8 data once in vectorization load/store does get optimized, code below
    * can begin with :
      if (address % vec8 == 0) {
        return std::min(4, valid_vec_size);
    */
  if (address % vec4 == 0) {
    return 4;
  } else if (address % vec2 == 0) {
    return 2;
  } else {
    return 1;
  }
}

template <typename func_t>
int GetVectorizedSize(const void* in, void* out) {
  int vec_size = 4;
  using traits = FunctionTraits<func_t>;
  // using ArgsT = typename traits::ArgsTuple;
  using arg1_t = typename traits::template arg<0>::type;
  // using return_t = typename traits::result_type;
  vec_size = std::min<int>(vec_size, GetVectorizedSize<arg1_t>(in));

  return vec_size;
}

template <int vec_size, typename func_t>
__device__ inline void single_ele_kernel_impl(
    int N, func_t f, const void* in, void* out_p, int tid) {
  using traits = FunctionTraits<func_t>;
  // using ArgsT = typename traits::ArgsTuple;
  using arg1_t = typename traits::template arg<0>::type;
  using return_t = typename traits::result_type;

  auto x = reinterpret_cast<const arg1_t*>(in);
  auto out = reinterpret_cast<return_t*>(out_p);

  out[tid] = f(x[tid]);
}

template <int vec_size, typename func_t>
__global__ void single_ele_kernel(int N, func_t f, const void* in, void* out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    single_ele_kernel_impl<vec_size, func_t>(N, f, in, out, tid);
  }
}

template <int vec_size, typename func_t>
__device__ void vectorize_kernel_impl(
    int N, func_t f, const void* in, void* out, int64_t tid) {
  using traits = FunctionTraits<func_t>;

  using arg1_t = typename traits::template arg<0>::type;
  using return_t = typename traits::result_type;

  using InVecType = AlignedVector<arg1_t, vec_size>;
  using OutVecType = AlignedVector<return_t, vec_size>;
  InVecType ins_vec[1];
  OutVecType out_vec;

  arg1_t* ins_ptr[1];
  arg1_t ins[1];

  ins_ptr[0] = reinterpret_cast<arg1_t*>(&(ins_vec[0]));

  // load
  const InVecType* in_vec_data = reinterpret_cast<const InVecType*>(in);
  ins_vec[0] = in_vec_data[tid];

// compute
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    out_vec.val[i] = f(ins_ptr[0][i]);
  }

  // store
  OutVecType* out_vec_ptr = reinterpret_cast<OutVecType*>(out);

  out_vec_ptr[tid] = out_vec;
}

template <int vec_size, typename func_t>
__global__ void vectorize_kernel(int N,
                                 func_t f,
                                 const void* in,
                                 void* out,
                                 int main_tid,
                                 int tail_tid,
                                 int64_t offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // printf( "%d\n", tid);
  if (tid < main_tid) {
    vectorize_kernel_impl<vec_size, func_t>(N, f, in, out, tid);
  }

  if (tid < tail_tid) {
    single_ele_kernel_impl<vec_size, func_t>(N, f, in, out, tid + offset);
  }
}

template <typename func_t>
static inline void launch_vectorized_kernel(
    gpuStream_t stream, int64_t N, const func_t& f, const void* in, void* out) {
  // TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  // using traits = function_traits<func_t>;

  int vec_size = GetVectorizedSize<func_t>(in, out);
  // std::cerr << "vec size " << GetVectorizedSize<func_t>( in, out) <<
  // std::endl;
  //  int vec_size = 1;
  int block_size = 64;
  int grid_size = ((N + vec_size - 1) / vec_size + block_size - 1) / block_size;

  int main_tid = N / vec_size;
  int tail_tid = N % vec_size;
  // std::cerr << "grid size "<< grid_size << "\t" << main_tid << "\t" <<
  // tail_tid << std::endl;

  switch (vec_size) {
    case 4: {
      vectorize_kernel<4, func_t><<<grid_size, block_size, 0, stream>>>(
          N, f, in, out, main_tid, tail_tid, main_tid * vec_size);
      break;
    }
    case 2: {
      vectorize_kernel<4, func_t><<<grid_size, block_size, 0, stream>>>(
          N, f, in, out, main_tid, tail_tid, main_tid * vec_size);
      break;
    }
    case 1: {
      single_ele_kernel<1, func_t>
          <<<grid_size, block_size, 0, stream>>>(N, f, in, out);
      break;
    }
    default: {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

template <typename func_t>
void test_func(gpuStream_t stream,
               const TensorContainer& tensor_con,
               const func_t& f) {
  const void* t1 = tensor_con.x_->data();
  void* t2 = tensor_con.out_->data();

  int num = tensor_con.x_->numel();

  launch_vectorized_kernel(stream, num, f, t1, t2);
}

}  // namespace funcs
}  // namespace phi
