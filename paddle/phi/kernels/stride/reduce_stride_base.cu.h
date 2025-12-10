// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"

namespace phi {

template <typename Context>
phi::DenseTensor Tensor2Contiguous(const Context& dev_ctx,
                                   const phi::DenseTensor& tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  PD_VISIT_ALL_TYPES(tensor.dtype(), "Tensor2Contiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

static inline int64_t DivUp(const int64_t& a, const int64_t& b) {
  return (a + b - 1) / b;
}

static inline int LastPow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

struct ReduceStrideConfig {
  static constexpr int BX = 0;
  static constexpr int BY = 1;
  static constexpr int GLO = 2;

  ReduceStrideConfig(int element_size_bytes, int num_outputs, int num_inputs)
      : element_size_bytes(element_size_bytes),
        num_inputs(num_inputs),
        num_outputs(num_outputs) {}
  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int reduce_per_output = 1;
  int input_tmp[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int b_w;
  int b_h;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;

  template <typename T>
  void set_block(int64_t dim0, int64_t dim1) {
    const int mx_threads = kps::details::kReduceMaxThread / output_vec_size;
    int dim0_pow2 =
        dim0 < mx_threads ? static_cast<int>(LastPow2(dim0)) : mx_threads;
    int dim1_pow2 =
        dim1 < mx_threads ? static_cast<int>(LastPow2(dim1)) : mx_threads;
    b_w = std::min(dim0_pow2, static_cast<int>(kps::details::kWarpSize));
    b_h = std::min(dim1_pow2, static_cast<int>(mx_threads / b_w));
    b_w = std::min(dim0_pow2, static_cast<int>(mx_threads / b_h));
    num_threads = b_w * b_h;
  }

  dim3 block() const { return dim3(b_w, b_h); }

  dim3 grid() const {
    return dim3(DivUp(num_outputs / output_vec_size, step_output),
                reduce_per_output);
  }

  __host__ __device__ bool check_x_reduce() const { return input_tmp[BX] != 0; }

  __host__ __device__ bool check_y_reduce() const { return input_tmp[BY] != 0; }

  __host__ __device__ bool enable_g_reduce() const {
    return input_tmp[GLO] != 0;
  }

  __device__ bool check_store(int output_idx) const {
    return output_idx < num_outputs &&
           (!check_x_reduce() || threadIdx.x == 0) &&
           (!check_y_reduce() || threadIdx.y == 0);
  }

  __device__ bool check_reduce_tail() const {
    return (!check_y_reduce() || threadIdx.y == 0) &&
           (!enable_g_reduce() || blockIdx.y == 0);
  }

  __host__ __device__ int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int reduce2 = blockIdx.y;
    return (lane * input_tmp[BX] + warp * input_tmp[BY] +
            reduce2 * input_tmp[GLO]);
  }

  template <int OUTPUT_VEC_SIZE>
  __host__ __device__ int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int reduce1 = blockIdx.x;
    return (lane * output_mult[BX] + warp * output_mult[BY] +
            reduce1 * step_output) *
           OUTPUT_VEC_SIZE;
  }

  __device__ int sm_off(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  __device__ int st_mem_off(int reduce2) const {
    int offset = reduce2 + blockIdx.x * gridDim.y;
    if (!check_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int sp_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int sp_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  int sm_size() const {
    if (!check_y_reduce() &&
        (!check_x_reduce() || b_w <= kps::details::kWarpSize)) {
      return 0;
    }
    return element_size_bytes * num_threads * output_vec_size;
  }

  int64_t gm_size() const {
    if (!enable_g_reduce()) {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * reduce_per_output;
    if (!check_x_reduce()) {
      size *= block().x * output_vec_size;
    }
    return size;
  }

  int sem_size() const {
    if (!enable_g_reduce()) {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  int value_pt() const { return DivUp(num_inputs, step_input); }
};

std::ostream& operator<<(std::ostream& out, const ReduceStrideConfig& config);

template <int nt, int OUTPUT_VEC_SIZE, typename R>
__global__ void reduce_kernel(R reduction) {
  reduction.template run<OUTPUT_VEC_SIZE>();
}

template <typename uint32_t>
static funcs::OffsetCalculator<2, uint32_t> make_output_calculator(
    const DenseTensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int num_output_dims = iter.ndim() - num_reduce_dims;
  int input_index = iter.ntensors() - 1;
  int output_index = 0;
  std::array<const int64_t*, 2> strides = {
      iter.strides(output_index).data() + num_reduce_dims,
      iter.strides(input_index).data() + num_reduce_dims,
  };
  auto shape = iter.shape().data() + num_reduce_dims;
  return funcs::OffsetCalculator<2, uint32_t>(
      num_output_dims, shape, strides.data());
}

template <typename uint32_t>
static funcs::OffsetCalculator<1, uint32_t> make_input_calculator(
    const DenseTensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int input_index = iter.ntensors() - 1;
  std::array<const int64_t*, 1> strides = {
      iter.strides(input_index).data(),
  };
  return funcs::OffsetCalculator<1, uint32_t>(
      num_reduce_dims, iter.shape().data(), strides.data());
}

template <typename T>
int get_outvec_size(const DenseTensorIterator& iter) {
  int vec_size = 4;
  auto update_outvec_size = [&vec_size](uint64_t n) {
    while (n % vec_size != 0) {
      vec_size /= 2;
    }
  };

  uint64_t base_address =
      reinterpret_cast<uint64_t>(iter.data_ptr(iter.noutputs())) / sizeof(T);
  update_outvec_size(base_address);

  const int output_index = iter.num_reduce_dims();
  update_outvec_size(iter.shape()[output_index]);

  int j = 0;
  for (auto i : iter.strides(iter.noutputs())) {
    if (j != output_index) {
      update_outvec_size(i / sizeof(T));
    }
    j++;
  }
  return vec_size;
}

template <typename T, int VALUE_VEC_SIZE, int INPUT_VEC_SIZE = VALUE_VEC_SIZE>
ReduceStrideConfig setReduceConfig(const DenseTensorIterator& iter) {
  int64_t num_outputs = iter.num_output_elements();
  int64_t inputs_per_output = iter.numel() / num_outputs;
  int input_index = iter.ntensors() - 1;

  auto config = ReduceStrideConfig(sizeof(T), num_outputs, inputs_per_output);

  int64_t dim0;
  int64_t dim1;
  int64_t fastest_moving_stride;
  bool reduction_on_fastest_striding_dimension;

  if (iter.ndim() > 0) {
    reduction_on_fastest_striding_dimension =
        (iter.num_reduce_dims() == iter.ndim()) ||
        (iter.strides(input_index)[0] <
         iter.strides(input_index)[iter.num_reduce_dims()]);
    if (reduction_on_fastest_striding_dimension) {
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = iter.strides(input_index)[0];
    } else {
      dim0 = num_outputs;
      dim1 = inputs_per_output;
      fastest_moving_stride = iter.strides(input_index)[iter.num_reduce_dims()];
    }
  } else {
    reduction_on_fastest_striding_dimension = true;
    fastest_moving_stride = sizeof(T);
    dim0 = 1;
    dim1 = 1;
  }
  if (fastest_moving_stride == sizeof(T)) {
    if (reduction_on_fastest_striding_dimension && dim0 > 128 &&
        iter.num_reduce_dims() == 1 && VALUE_VEC_SIZE >= INPUT_VEC_SIZE) {
      config.vectorize_input = true;
      dim0 /= INPUT_VEC_SIZE;
    } else if (!reduction_on_fastest_striding_dimension) {
      config.output_vec_size = get_outvec_size<T>(iter);
      dim0 /= config.output_vec_size;
    }
  }

  config.set_block<T>(dim0, dim1);

  int b_w = config.b_w;
  int b_h = config.b_h;

  if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
    config.input_tmp[0] = config.sp_input(b_w);
  } else {
    config.output_mult[0] = config.sp_output(b_w);
  }

  constexpr int min_values_per_thread = 16;
  constexpr int max_values_per_thread = 256;

  int device_id = phi::backends::gpu::GetCurrentDeviceId();

  const int warp_split_threshold =
      std::min<int>(b_h * 16, max_values_per_thread);
  bool split_across_warps = config.value_pt() >= warp_split_threshold;
  const int num_mp = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  if (split_across_warps) {
    config.input_tmp[1] = config.sp_input(b_h);
  } else {
    config.output_mult[1] = config.sp_output(b_h);
  }

  int max_threads_per_mp =
      phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(device_id);

  const int blocks_per_sm = max_threads_per_mp / config.num_threads;
  const int target_grid_size = num_mp * blocks_per_sm;
  int grid = config.grid().x;
  if (config.input_tmp[1] != 0 && config.value_pt() >= max_values_per_thread &&
      grid <= target_grid_size) {
    int reduce_per_output1 = DivUp(target_grid_size, grid);
    int reduce_per_output2 = DivUp(config.value_pt(), min_values_per_thread);
    int reduce_per_output3 = DivUp(config.value_pt(), max_values_per_thread);
    config.reduce_per_output =
        std::max(std::min<int>(reduce_per_output1, reduce_per_output2),
                 reduce_per_output3);
    if (config.reduce_per_output > 1) {
      config.input_tmp[2] = config.sp_input(config.reduce_per_output);
    }
  }
  return config;
}

template <typename T, int NX, int NY, bool IsBoundary = false>
__device__ __forceinline__ void VecReadData(T* dst, const T* __restrict__ src) {
  if (IsBoundary) {
    int64_t thread_offset = 0;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < NX) {
        dst[idx] = src[thread_offset + idx];
      }
    }
  } else {
    constexpr int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    constexpr int kVectorsPerThread = NX / kVectorSize;

    using VecType = kps::details::VectorType<T, kVectorSize>;
    const VecType* vec_input = reinterpret_cast<const VecType*>(src);
    VecType vec_temp[kVectorsPerThread];

#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
      vec_temp[i] = vec_input[i];
#pragma unroll
      for (int idx = 0; idx < NX; ++idx) {
        dst[idx] = *(reinterpret_cast<T*>(vec_temp) + idx);
      }
    }
  }
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ T InterWarpReduce(T val, ReduceOp reducer) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  // hack WarpSize = 32 to pass ROCM unittest
  for (int stride = 32 / 2; stride > 0; stride >>= 1) {
    T temp = phi::backends::gpu::CudaShuffleDownSync(mask, val, stride);
    val = reducer(val, temp);
  }
  return val;
}

template <typename T,
          typename OP_T,
          int VALUE_VEC_SIZE = 4,
          int INPUT_VEC_SIZE = VALUE_VEC_SIZE>
struct ReduceStrideOp {
  using InputCalculator = funcs::OffsetCalculator<1, uint32_t>;
  using OutputCalculator = funcs::OffsetCalculator<2, uint32_t>;

  OP_T ops;
  T ident;
  ReduceStrideConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void* src;
  char* dst;
  void* red_buf;
  int* sem;
  int noutputs;
  bool is_mean;
  int64_t mean_factor;

  ReduceStrideOp(OP_T ops,
                 ReduceStrideConfig config,
                 InputCalculator input_calc,
                 OutputCalculator output_calc,
                 const void* src,
                 char* dst0,
                 void* red_buf,
                 int* sem,
                 T ident,
                 int noutputs,
                 bool is_mean,
                 int64_t mean_factor)
      : ops(ops),
        ident(ident),
        config(config),
        input_calc(input_calc),
        output_calc(output_calc),
        src(src),
        red_buf(red_buf),
        sem(sem),
        noutputs(noutputs),
        is_mean(is_mean),
        mean_factor(mean_factor) {
    dst = dst0;
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ void run() const {
    extern __shared__ char share_mem[];
    uint32_t output_idx = config.output_idx<OUTPUT_VEC_SIZE>();
    uint32_t input_idx = config.input_idx();
    auto base_off = output_calc.get(output_idx)[1];
    using ARG_VEC_T = std::array<T, OUTPUT_VEC_SIZE>;
    ARG_VEC_T value;

    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      const T* input_off = (const T*)((const char*)src + base_off);
      value = th_reduce<OUTPUT_VEC_SIZE>(input_off);
    }
    if (config.check_y_reduce()) {
      value = by_reduce<OUTPUT_VEC_SIZE>(value, share_mem);
    }
    if (config.check_x_reduce()) {
      value = bx_reduce<OUTPUT_VEC_SIZE>(value, share_mem);
    }

    using OUT_VEC_T = std::array<T*, OUTPUT_VEC_SIZE>;
    using OFF_VEC_T = std::array<uint32_t, OUTPUT_VEC_SIZE>;
    OFF_VEC_T base_offsets;
    OUT_VEC_T out;

#pragma unroll
    for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] =
          reinterpret_cast<T*>(reinterpret_cast<char*>(dst) + base_offsets[i]);
    }

    if (config.enable_g_reduce()) {
      value = global_reduce<OUTPUT_VEC_SIZE>(value, share_mem);
    } else if (config.check_store(output_idx)) {
#pragma unroll
      for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
        if (is_mean) {
          value[i] = value[i] / static_cast<T>(mean_factor);
        }
        *(out[i]) = value[i];
      }
    }
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ std::array<T, OUTPUT_VEC_SIZE> th_reduce(const T* data) const {
    if (config.vectorize_input) {
      return {inputvec_th_reduce(data)};
    } else {
      uint32_t element_stride = input_calc.strides_[0][0] / sizeof(T);
      bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
      if (is_contiguous) {
        return th_reduce_impl<OUTPUT_VEC_SIZE>(
            data, [](uint32_t idx) { return idx; });
      } else if (input_calc.dims == 1) {
        return th_reduce_impl<OUTPUT_VEC_SIZE>(
            data, [&](uint32_t idx) { return idx * element_stride; });
      } else {
        return th_reduce_impl<OUTPUT_VEC_SIZE>(data, [&](uint32_t idx) {
          return input_calc.get(idx)[0] / sizeof(T);
        });
      }
    }
  }

  __device__ T inputvec_th_reduce(const T* data) const {
    uint32_t end = config.num_inputs;
    T value = ident;
    constexpr int align_bytes = INPUT_VEC_SIZE * sizeof(T);
    constexpr int align_elements = align_bytes / sizeof(T);
    int shift = ((uint64_t)data) % align_bytes / sizeof(T);

    if (shift > 0) {
      data -= shift;
      end += shift;
      if (threadIdx.x >= shift && threadIdx.x < align_elements &&
          config.check_reduce_tail()) {
        T tmp_value;
        kps::details::ReadData<T>(
            &tmp_value,
            reinterpret_cast<const T*>(data + threadIdx.x),
            INPUT_VEC_SIZE);
        kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
            &value, &tmp_value, ops, false);
      }
      end -= align_elements;
      data += align_elements;
      shift = align_elements - shift;
    }

    uint32_t idx = config.input_idx();
    const uint32_t stride = config.step_input;

    T value_[INPUT_VEC_SIZE];
    value_[0] = value;

#pragma unroll
    for (int i = 1; i < INPUT_VEC_SIZE; i++) {
      value_[i] = ident;
    }

    while (idx * INPUT_VEC_SIZE + INPUT_VEC_SIZE - 1 < end) {
      T input_vec[INPUT_VEC_SIZE];
      VecReadData<T, INPUT_VEC_SIZE, 1, false>(
          &(input_vec[0]),
          reinterpret_cast<const T*>(data + idx * INPUT_VEC_SIZE));

#pragma unroll
      for (uint32_t i = 0; i < INPUT_VEC_SIZE; i++) {
        kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
            &(value_[i]), &(input_vec[i]), ops, false);
      }

      idx += stride;
    }

    uint32_t tail_start = end - end % INPUT_VEC_SIZE;
    if (config.check_reduce_tail()) {
      int idx = tail_start + threadIdx.x;
      if (idx < end) {
        T value;
        kps::details::ReadData<T>(
            &value, reinterpret_cast<const T*>(data + idx), 1);
        kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
            &(value_[0]), &value, ops, false);
      }
    }

#pragma unroll
    for (int i = 1; i < INPUT_VEC_SIZE; i++) {
      kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
          &(value_[0]), &(value_[i]), ops, false);
    }

    return value_[0];
  }

  template <int OUTPUT_VEC_SIZE, typename OFFCALC_T>
  __device__ std::array<T, OUTPUT_VEC_SIZE> th_reduce_impl(
      const T* data_, OFFCALC_T offset_calc) const {
    uint32_t idx = config.input_idx();
    const uint32_t end = config.num_inputs;
    const uint32_t stride = config.step_input;

    using ARG_VEC_T = std::array<T, OUTPUT_VEC_SIZE>;

    ARG_VEC_T value_[VALUE_VEC_SIZE];

#pragma unroll
    for (int i = 0; i < VALUE_VEC_SIZE; i++) {
#pragma unroll
      for (int j = 0; j < OUTPUT_VEC_SIZE; j++) {
        value_[i][j] = ident;
      }
    }

    T values[VALUE_VEC_SIZE];

    while (idx + (VALUE_VEC_SIZE - 1) * stride < end) {
#pragma unroll
      for (uint32_t i = 0; i < VALUE_VEC_SIZE; i++) {
        const auto offset = offset_calc(idx + i * stride) / OUTPUT_VEC_SIZE;
        kps::details::ReadData<T>(&(values[i]),
                                  reinterpret_cast<const T*>(data_ + offset),
                                  VALUE_VEC_SIZE);
      }
#pragma unroll
      for (uint32_t i = 0; i < VALUE_VEC_SIZE; i++) {
#pragma unroll
        for (uint32_t j = 0; j < OUTPUT_VEC_SIZE; j++) {
          kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
              &(value_[i][j]), &(values[i]), ops, false);
        }
      }
      idx += stride * VALUE_VEC_SIZE;
    }

    int idx_ = idx;
#pragma unroll
    for (uint32_t i = 0; i < VALUE_VEC_SIZE; i++) {
      if (idx >= end) {
        break;
      }
      const auto offset = offset_calc(idx) / OUTPUT_VEC_SIZE;
      kps::details::ReadData<T>(&(values[i]),
                                reinterpret_cast<const T*>(data_ + offset),
                                VALUE_VEC_SIZE);
      idx += stride;
    }
    idx = idx_;
#pragma unroll
    for (uint32_t i = 0; i < VALUE_VEC_SIZE; i++) {
      if (idx >= end) {
        break;
      }
#pragma unroll
      for (uint32_t j = 0; j < OUTPUT_VEC_SIZE; j++) {
        kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
            &(value_[i][j]), &(values[i]), ops, false);
      }
      idx += stride;
    }

#pragma unroll
    for (int i = 1; i < VALUE_VEC_SIZE; i++) {
#pragma unroll
      for (uint32_t j = 0; j < OUTPUT_VEC_SIZE; j++) {
        kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
            &(value_[0][j]), &(value_[i][j]), ops, false);
      }
    }
    return value_[0];
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ std::array<T, OUTPUT_VEC_SIZE> bx_reduce(
      std::array<T, OUTPUT_VEC_SIZE> value, char* share_mem) const {
    using ARG_VEC_T = std::array<T, OUTPUT_VEC_SIZE>;
    int dim_x = blockDim.x;
    ARG_VEC_T* shared = reinterpret_cast<ARG_VEC_T*>(share_mem);
    if (dim_x > kps::details::kWarpSize) {
      int address_base = threadIdx.x + threadIdx.y * blockDim.x;
      shared[address_base] = value;
      for (int offset = dim_x / 2; offset >= kps::details::kWarpSize;
           offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
          ARG_VEC_T other = shared[address_base + offset];
#pragma unroll
          for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
            kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
                &(value[i]), &(other[i]), ops, false);
          }
          shared[address_base] = value;
        }
      }
      dim_x = kps::details::kWarpSize;
    }

    __syncthreads();
    value[0] = InterWarpReduce<T, OP_T>(value[0], ops);

    return value;
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ std::array<T, OUTPUT_VEC_SIZE> by_reduce(
      std::array<T, OUTPUT_VEC_SIZE> value, char* share_mem) const {
    using ARG_VEC_T = std::array<T, OUTPUT_VEC_SIZE>;
    ARG_VEC_T* shared = reinterpret_cast<ARG_VEC_T*>(share_mem);
    shared[config.sm_off(0)] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
        ARG_VEC_T other = shared[config.sm_off(offset)];
#pragma unroll
        for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
          kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
              &(value[i]), &(other[i]), ops, false);
        }
        shared[config.sm_off(0)] = value;
      }
    }
    return value;
  }

  __device__ bool check_finish() const {
    __shared__ bool is_done;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev_blocks_finished = atomicAdd(&sem[blockIdx.x], 1);
      is_done = (prev_blocks_finished == gridDim.y - 1);
    }

    __syncthreads();

    return is_done;
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ std::array<T, OUTPUT_VEC_SIZE> global_reduce(
      std::array<T, OUTPUT_VEC_SIZE> value, char* share_mem) const {
    using ARG_VEC_T = std::array<T, OUTPUT_VEC_SIZE>;
    using OUT_VEC_T = std::array<T*, OUTPUT_VEC_SIZE>;
    using OFF_VEC_T = std::array<uint32_t, OUTPUT_VEC_SIZE>;

    ARG_VEC_T* reduce_buffer = reinterpret_cast<ARG_VEC_T*>(red_buf);
    uint32_t output_idx = config.output_idx<OUTPUT_VEC_SIZE>();
    OFF_VEC_T base_offsets;
    OUT_VEC_T out;

#pragma unroll
    for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] =
          reinterpret_cast<T*>(reinterpret_cast<char*>(dst) + base_offsets[i]);
    }

    bool check_store = config.check_store(output_idx);
    if (check_store) {
      uint32_t offset = config.st_mem_off(blockIdx.y);
      reduce_buffer[offset] = value;
    }

    __threadfence();
    __syncthreads();
    bool is_last_block_done = check_finish();

    if (is_last_block_done) {
      __threadfence();
      for (auto& v : value) {
        v = ident;
      }
      if (config.check_x_reduce()) {
        uint32_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
        uint32_t step = blockDim.x * blockDim.y;
        for (; input_offset < config.reduce_per_output; input_offset += step) {
          uint32_t idx = config.st_mem_off(input_offset);
          ARG_VEC_T next = reduce_buffer[idx];
#pragma unroll
          for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
            kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
                &(value[i]), &(next[i]), ops, false);
          }
        }
      } else {
        uint32_t input_offset = threadIdx.y;
        uint32_t step = blockDim.y;
        for (; input_offset < config.reduce_per_output; input_offset += step) {
          uint32_t idx = config.st_mem_off(input_offset);
          ARG_VEC_T next = reduce_buffer[idx];
#pragma unroll
          for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
            kps::Reduce<T, 1, 1, OP_T, kps::details::ReduceMode::kLocalMode>(
                &(value[i]), &(next[i]), ops, false);
          }
        }
      }

      value = by_reduce<OUTPUT_VEC_SIZE>(value, share_mem);

      if (config.check_x_reduce()) {
        value = bx_reduce<OUTPUT_VEC_SIZE>(value, share_mem);
      }

      if (check_store) {
#pragma unroll
        for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
          if (is_mean) {
            value[i] = value[i] / static_cast<T>(mean_factor);
          }
          *(out[i]) = value[i];
        }
      }
    }

    return value;
  }
};

template <typename Context, int max_threads, typename R>
static void LaunchReduceStride(const Context& dev_ctx,
                               const ReduceStrideConfig& config,
                               const R& reduction) {
  dim3 block = config.block();
  dim3 grid = config.grid();

  int share_mem = config.sm_size();
  auto stream = dev_ctx.stream();
  reduce_kernel<max_threads / 1, 1, R>
      <<<grid, block, share_mem, stream>>>(reduction);
}

// TODO(wangjinheng): Support Multi-Dim Reduction

template <typename T,
          typename Context,
          template <typename>
          class reduce_op,
          bool IsMean = false>
void ReduceStrideImpl(const Context& dev_ctx,
                      const DenseTensor& x,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      T ident,
                      DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  DenseTensorIteratorConfig config;
  config.is_reduction(true);
  config.add_output(*(out));
  config.add_const_input(x);
  DenseTensorIterator iter = config.build();

  const char* in_data =
      reinterpret_cast<const char*>(iter.data_ptr(iter.ntensors() - 1));
  char* out_data = reinterpret_cast<char*>(out->data<T>());
  const auto noutputs = iter.noutputs();

  constexpr int VALUE_VEC_SIZE = 4;
  constexpr int INPUT_VEC_SIZE = 4;

  ReduceStrideConfig reduce_stride_conf =
      setReduceConfig<T, VALUE_VEC_SIZE>(iter);

  void* reduce_buf;
  void* reduce_sem;

  DenseTensor reduce_buf_tensor;
  DenseTensor reduce_sem_tensor;

  PADDLE_ENFORCE_LT(
      reduce_stride_conf.gm_size(),
      std::numeric_limits<int32_t>::max(),
      ::common::errors::InvalidArgument(
          "reduce_stride_conf.gm_size() should be less than int32_t"));

  PADDLE_ENFORCE_LT(
      reduce_stride_conf.sem_size(),
      std::numeric_limits<int32_t>::max(),
      ::common::errors::InvalidArgument(
          "reduce_stride_conf.sem_size() should be less than int32_t"));

  std::vector<int> reduce_buf_size = {
      static_cast<int>(reduce_stride_conf.gm_size() / phi::SizeOf(x.dtype()))};
  std::vector<int> reduce_sem_size = {
      static_cast<int>(reduce_stride_conf.sem_size() / phi::SizeOf(x.dtype()))};

  if (reduce_buf_size[0] == 0) {
    reduce_buf_size[0] = phi::SizeOf(x.dtype());
  }

  if (reduce_sem_size[0] == 0) {
    reduce_sem_size[0] = phi::SizeOf(x.dtype());
  }

  if (reduce_stride_conf.enable_g_reduce()) {
    reduce_buf_tensor.Resize(common::make_ddim(reduce_buf_size));
    reduce_sem_tensor.Resize(common::make_ddim(reduce_sem_size));

    reduce_buf =
        reinterpret_cast<void*>(dev_ctx.template Alloc<T>(&reduce_buf_tensor));
    reduce_sem =
        reinterpret_cast<void*>(dev_ctx.template Alloc<T>(&reduce_sem_tensor));

    auto stream = dev_ctx.stream();
    phi::backends::gpu::GpuMemsetAsync(
        reduce_sem, 0, reduce_stride_conf.sem_size(), stream);
  }

  auto output_calc = make_output_calculator<uint32_t>(iter);
  auto input_calc = make_input_calculator<uint32_t>(iter);

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  auto reducer = reduce_op<MPType>();

  int64_t mean_factor = iter.numel();

  auto reduce =
      ReduceStrideOp<T, reduce_op<MPType>, VALUE_VEC_SIZE, INPUT_VEC_SIZE>(
          reducer,
          reduce_stride_conf,
          input_calc,
          output_calc,
          in_data,
          out_data,
          reduce_buf,
          reinterpret_cast<int*>(reduce_sem),
          ident,
          noutputs,
          IsMean,
          mean_factor);
  constexpr int MaxThread = kps::details::kReduceMaxThread;

  LaunchReduceStride<Context, MaxThread>(dev_ctx, reduce_stride_conf, reduce);
}

}  // namespace phi

#endif
