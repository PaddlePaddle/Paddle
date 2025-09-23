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

#if defined(__NVCC__)

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
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  ReduceStrideConfig(int element_size_bytes, int num_outputs, int num_inputs)
      : element_size_bytes(element_size_bytes),
        num_inputs(num_inputs),
        num_outputs(num_outputs) {}
  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;

  int warp_size = 32;

  int MAX_THREADS = 512;

  template <typename T>
  void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_num_threads = MAX_THREADS / output_vec_size;
    int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(LastPow2(dim0))
                                           : max_num_threads;
    int dim1_pow2 = dim1 < max_num_threads ? static_cast<int>(LastPow2(dim1))
                                           : max_num_threads;
    block_width = std::min(dim0_pow2, static_cast<int>(warp_size));
    block_height =
        std::min(dim1_pow2, static_cast<int>(max_num_threads / block_width));
    block_width =
        std::min(dim0_pow2, static_cast<int>(max_num_threads / block_height));
    num_threads = block_width * block_height;
  }

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  dim3 block() const { return dim3(block_width, block_height); }

  dim3 grid() const {
    return dim3(DivUp(num_outputs / output_vec_size, step_output),
                ctas_per_output);
  }

  __host__ __device__ bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  __host__ __device__ bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  __host__ __device__ bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  __device__ bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
           (!should_block_x_reduce() || threadIdx.x == 0) &&
           (!should_block_y_reduce() || threadIdx.y == 0);
  }

  __device__ bool should_reduce_tail() const {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
           (!should_global_reduce() || blockIdx.y == 0);
  }

  __host__ __device__ int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] + warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  template <int OUTPUT_VEC_SIZE>
  __host__ __device__ int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] + warp * output_mult[BLOCK_Y] +
            cta1 * step_output) *
           OUTPUT_VEC_SIZE;
  }

  __device__ int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  __device__ int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() || block_width <= warp_size)) {
      return 0;
    }
    return element_size_bytes * num_threads * output_vec_size;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce()) {
      size *= block().x * output_vec_size;
    }
    return size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  int values_per_thread() const { return DivUp(num_inputs, step_input); }
};

std::ostream& operator<<(std::ostream& out, const ReduceStrideConfig& config);

template <int nt, int OUTPUT_VEC_SIZE, typename R>
__global__ void reduce_kernel(R reduction) {
  reduction.template run<OUTPUT_VEC_SIZE>();
}

template <typename index_t>
static funcs::OffsetCalculator<2, index_t> make_output_calculator(
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
  return funcs::OffsetCalculator<2, index_t>(
      num_output_dims, shape, strides.data());
}

template <typename index_t>
static funcs::OffsetCalculator<1, index_t> make_input_calculator(
    const DenseTensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int input_index = iter.ntensors() - 1;
  std::array<const int64_t*, 1> strides = {
      iter.strides(input_index).data(),
  };
  return funcs::OffsetCalculator<1, index_t>(
      num_reduce_dims, iter.shape().data(), strides.data());
}

template <typename T>
int get_output_vec_size(const DenseTensorIterator& iter) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](uint64_t n) {
    while (n % vec_size != 0) {
      vec_size /= 2;
    }
  };

  uint64_t base_address =
      reinterpret_cast<uint64_t>(iter.data_ptr(iter.noutputs())) / sizeof(T);
  update_vec_size(base_address);

  const int output_index = iter.num_reduce_dims();
  update_vec_size(iter.shape()[output_index]);

  int j = 0;
  for (auto i : iter.strides(iter.noutputs())) {
    if (j != output_index) {
      update_vec_size(i / sizeof(T));
    }
    j++;
  }
  return vec_size;
}

template <typename arg_t, typename scalar_t, int VT0, int INPUT_VEC_SIZE = VT0>
ReduceStrideConfig setReduceConfig(const DenseTensorIterator& iter) {
  int64_t num_outputs = iter.num_output_elements();
  int64_t inputs_per_output = iter.numel() / num_outputs;
  int input_index = iter.ntensors() - 1;

  auto config =
      ReduceStrideConfig(sizeof(arg_t), num_outputs, inputs_per_output);

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
    fastest_moving_stride = sizeof(scalar_t);
    dim0 = 1;
    dim1 = 1;
  }
  if (fastest_moving_stride == sizeof(scalar_t)) {
    if (reduction_on_fastest_striding_dimension && dim0 > 128 &&
        iter.num_reduce_dims() == 1 && VT0 >= INPUT_VEC_SIZE) {
      config.vectorize_input = true;
      dim0 /= INPUT_VEC_SIZE;
    } else if (!reduction_on_fastest_striding_dimension) {
      config.output_vec_size = get_output_vec_size<scalar_t>(iter);
      dim0 /= config.output_vec_size;
    }
  }

  config.set_block_dimension<scalar_t>(dim0, dim1);

  int block_width = config.block_width;
  int block_height = config.block_height;

  if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
    config.input_mult[0] = config.split_input(block_width);
  } else {
    config.output_mult[0] = config.split_output(block_width);
  }

  constexpr int min_values_per_thread = 16;
  constexpr int max_values_per_thread = 256;

  int device_id = phi::backends::gpu::GetCurrentDeviceId();

  const int warp_split_threshold =
      std::min<int>(block_height * 16, max_values_per_thread);
  bool split_across_warps = config.values_per_thread() >= warp_split_threshold;
  const int num_mp = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  if (split_across_warps) {
    config.input_mult[1] = config.split_input(block_height);
  } else {
    config.output_mult[1] = config.split_output(block_height);
  }

  int max_threads_per_mp =
      phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(device_id);
  const int blocks_per_sm = max_threads_per_mp / config.num_threads;
  const int target_grid_size = num_mp * blocks_per_sm;
  int grid = config.grid().x;
  if (config.input_mult[1] != 0 &&
      config.values_per_thread() >= max_values_per_thread &&
      grid <= target_grid_size) {
    int ctas_per_output1 = DivUp(target_grid_size, grid);
    int ctas_per_output2 =
        DivUp(config.values_per_thread(), min_values_per_thread);
    int ctas_per_output3 =
        DivUp(config.values_per_thread(), max_values_per_thread);
    config.ctas_per_output = std::max(
        std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);
    if (config.ctas_per_output > 1) {
      config.input_mult[2] = config.split_input(config.ctas_per_output);
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

template <typename scalar_t,
          typename ops_t,
          typename index_t,
          typename out_scalar_t = scalar_t,
          int VT0 = 4,
          int INPUT_VEC_SIZE = VT0>
struct ReduceStrideOp {
  using arg_t = scalar_t;

  using InputCalculator = funcs::OffsetCalculator<1, index_t>;
  using OutputCalculator = funcs::OffsetCalculator<2, index_t>;

  static constexpr bool can_accumulate_in_output =
      std::is_convertible_v<arg_t, out_scalar_t> &&
      std::is_convertible_v<out_scalar_t, arg_t>;

  ops_t ops;
  arg_t ident;
  ReduceStrideConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void* src;
  char* dst[2];
  void* cta_buf;
  int* semaphores;
  int64_t base_idx;
  bool accumulate;
  bool final_output;
  int noutputs;
  bool is_mean;
  int64_t mean_factor;

  ReduceStrideOp(ops_t ops,
                 ReduceStrideConfig config,
                 InputCalculator input_calc,
                 OutputCalculator output_calc,
                 const void* src,
                 char* dst0,
                 void* cta_buf,
                 int* semaphores,
                 arg_t ident,
                 int noutputs,
                 int64_t base_idx,
                 bool is_mean,
                 int64_t mean_factor)
      : ops(ops),
        ident(ident),
        config(config),
        input_calc(input_calc),
        output_calc(output_calc),
        src(src),
        cta_buf(cta_buf),
        semaphores(semaphores),
        base_idx(base_idx),
        noutputs(noutputs),
        is_mean(is_mean),
        mean_factor(mean_factor) {
    dst[0] = dst0;
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ void run() const {
    extern __shared__ char shared_memory[];
    index_t output_idx = config.output_idx<OUTPUT_VEC_SIZE>();
    index_t input_idx = config.input_idx();
    auto base_offsets1 = output_calc.get(output_idx)[1];
    using arg_vec_t = std::array<arg_t, OUTPUT_VEC_SIZE>;
    arg_vec_t value;

    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      const scalar_t* input_slice =
          (const scalar_t*)((const char*)src + base_offsets1);
      value = thread_reduce<OUTPUT_VEC_SIZE>(input_slice);
    }
    if (config.should_block_y_reduce()) {
      value = block_y_reduce<OUTPUT_VEC_SIZE>(value, shared_memory);
    }
    if (config.should_block_x_reduce()) {
      value = block_x_reduce<OUTPUT_VEC_SIZE>(value, shared_memory);
    }

    using out_ptr_vec_t = std::array<out_scalar_t*, OUTPUT_VEC_SIZE>;
    using offset_vec_t = std::array<index_t, OUTPUT_VEC_SIZE>;
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

#pragma unroll
    for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = reinterpret_cast<out_scalar_t*>(reinterpret_cast<char*>(dst[0]) +
                                               base_offsets[i]);
    }

    if (config.should_global_reduce()) {
      value = global_reduce<OUTPUT_VEC_SIZE>(value, shared_memory);
    } else if (config.should_store(output_idx)) {
#pragma unroll
      for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
        if (is_mean) {
          value[i] = value[i] / static_cast<arg_t>(mean_factor);
        }
        *(out[i]) =
            get_accumulated_output<can_accumulate_in_output>(out[i], value[i]);
      }
    }
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ std::array<arg_t, OUTPUT_VEC_SIZE> thread_reduce(
      const scalar_t* data) const {
    if (config.vectorize_input) {
      return {input_vectorized_thread_reduce_impl(data)};
    } else {
      index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
      bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
      if (is_contiguous) {
        return thread_reduce_impl<OUTPUT_VEC_SIZE>(
            data, [](index_t idx) { return idx; });
      } else if (input_calc.dims == 1) {
        return thread_reduce_impl<OUTPUT_VEC_SIZE>(
            data, [&](index_t idx) { return idx * element_stride; });
      } else {
        return thread_reduce_impl<OUTPUT_VEC_SIZE>(data, [&](index_t idx) {
          return input_calc.get(idx)[0] / sizeof(scalar_t);
        });
      }
    }
  }

  __device__ arg_t
  input_vectorized_thread_reduce_impl(const scalar_t* data) const {
    index_t end = config.num_inputs;
    arg_t value = ident;
    constexpr int align_bytes = INPUT_VEC_SIZE * sizeof(scalar_t);
    constexpr int align_elements = align_bytes / sizeof(scalar_t);
    int shift = ((uint64_t)data) % align_bytes / sizeof(scalar_t);

    if (shift > 0) {
      data -= shift;
      end += shift;
      if (threadIdx.x >= shift && threadIdx.x < align_elements &&
          config.should_reduce_tail()) {
        arg_t tmp_value;
        kps::details::ReadData<arg_t>(
            &tmp_value,
            reinterpret_cast<const arg_t*>(data + threadIdx.x),
            INPUT_VEC_SIZE);
        kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
            &value, &tmp_value, ops, false);
      }
      end -= align_elements;
      data += align_elements;
      shift = align_elements - shift;
    }

    index_t idx = config.input_idx();
    const index_t stride = config.step_input;

    arg_t value_list[INPUT_VEC_SIZE];
    value_list[0] = value;

#pragma unroll
    for (int i = 1; i < INPUT_VEC_SIZE; i++) {
      value_list[i] = ident;
    }

    while (idx * INPUT_VEC_SIZE + INPUT_VEC_SIZE - 1 < end) {
      arg_t values_vec[INPUT_VEC_SIZE];
      VecReadData<arg_t, INPUT_VEC_SIZE, 1, false>(
          &(values_vec[0]),
          reinterpret_cast<const arg_t*>(data + idx * INPUT_VEC_SIZE));

#pragma unroll
      for (index_t i = 0; i < INPUT_VEC_SIZE; i++) {
        kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
            &(value_list[i]), &(values_vec[i]), ops, false);
      }

      idx += stride;
    }

    index_t tail_start = end - end % INPUT_VEC_SIZE;
    if (config.should_reduce_tail()) {
      int idx = tail_start + threadIdx.x;
      if (idx < end) {
        arg_t value;
        kps::details::ReadData<arg_t>(
            &value, reinterpret_cast<const arg_t*>(data + idx), 1);
        kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
            &(value_list[0]), &value, ops, false);
      }
    }

#pragma unroll
    for (int i = 1; i < INPUT_VEC_SIZE; i++) {
      kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
          &(value_list[0]), &(value_list[i]), ops, false);
    }

    return value_list[0];
  }

  template <int OUTPUT_VEC_SIZE, typename offset_calc_t>
  __device__ std::array<arg_t, OUTPUT_VEC_SIZE> thread_reduce_impl(
      const scalar_t* data_, offset_calc_t calc) const {
    index_t idx = config.input_idx();
    const index_t end = config.num_inputs;
    const index_t stride = config.step_input;

    using arg_vec_t = std::array<arg_t, OUTPUT_VEC_SIZE>;

    arg_vec_t value_list[VT0];

#pragma unroll
    for (int i = 0; i < VT0; i++) {
#pragma unroll
      for (int j = 0; j < OUTPUT_VEC_SIZE; j++) {
        value_list[i][j] = ident;
      }
    }

    arg_t values[VT0];

    while (idx + (VT0 - 1) * stride < end) {
#pragma unroll
      for (index_t i = 0; i < VT0; i++) {
        const auto offset = calc(idx + i * stride) / OUTPUT_VEC_SIZE;
        kps::details::ReadData<arg_t>(
            &(values[i]), reinterpret_cast<const arg_t*>(data_ + offset), VT0);
      }
#pragma unroll
      for (index_t i = 0; i < VT0; i++) {
#pragma unroll
        for (index_t j = 0; j < OUTPUT_VEC_SIZE; j++) {
          kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
              &(value_list[i][j]), &(values[i]), ops, false);
        }
      }
      idx += stride * VT0;
    }

    int idx_ = idx;
#pragma unroll
    for (index_t i = 0; i < VT0; i++) {
      if (idx >= end) {
        break;
      }
      const auto offset = calc(idx) / OUTPUT_VEC_SIZE;
      kps::details::ReadData<arg_t>(
          &(values[i]), reinterpret_cast<const arg_t*>(data_ + offset), VT0);
      idx += stride;
    }
    idx = idx_;
#pragma unroll
    for (index_t i = 0; i < VT0; i++) {
      if (idx >= end) {
        break;
      }
#pragma unroll
      for (index_t j = 0; j < OUTPUT_VEC_SIZE; j++) {
        kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
            &(value_list[i][j]), &(values[i]), ops, false);
      }
      idx += stride;
    }

#pragma unroll
    for (int i = 1; i < VT0; i++) {
#pragma unroll
      for (index_t j = 0; j < OUTPUT_VEC_SIZE; j++) {
        kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
            &(value_list[0][j]), &(value_list[i][j]), ops, false);
      }
    }
    return value_list[0];
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ std::array<arg_t, OUTPUT_VEC_SIZE> block_x_reduce(
      std::array<arg_t, OUTPUT_VEC_SIZE> value, char* shared_memory) const {
    using args_vec_t = std::array<arg_t, OUTPUT_VEC_SIZE>;
    int dim_x = blockDim.x;
    args_vec_t* shared = reinterpret_cast<args_vec_t*>(shared_memory);
    if (dim_x > warpSize) {
      int address_base = threadIdx.x + threadIdx.y * blockDim.x;
      shared[address_base] = value;
      for (int offset = dim_x / 2; offset >= warpSize; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
          args_vec_t other = shared[address_base + offset];
#pragma unroll
          for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
            kps::Reduce<arg_t,
                        1,
                        1,
                        ops_t,
                        kps::details::ReduceMode::kLocalMode>(
                &(value[i]), &(other[i]), ops, false);
          }
          shared[address_base] = value;
        }
      }
      dim_x = warpSize;
    }

    __syncthreads();

    value[0] = kps::details::WarpReduce<arg_t, ops_t>(value[0], ops);

    return value;
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ std::array<arg_t, OUTPUT_VEC_SIZE> block_y_reduce(
      std::array<arg_t, OUTPUT_VEC_SIZE> value, char* shared_memory) const {
    using args_vec_t = std::array<arg_t, OUTPUT_VEC_SIZE>;
    args_vec_t* shared = reinterpret_cast<args_vec_t*>(shared_memory);
    shared[config.shared_memory_offset(0)] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
        args_vec_t other = shared[config.shared_memory_offset(offset)];
#pragma unroll
        for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
          kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
              &(value[i]), &(other[i]), ops, false);
        }
        shared[config.shared_memory_offset(0)] = value;
      }
    }
    return value;
  }

  __device__ bool mark_block_finished() const {
    __shared__ bool is_last_block_done_shared;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
    }

    __syncthreads();

    return is_last_block_done_shared;
  }

  template <int OUTPUT_VEC_SIZE, bool can_acc>
  __device__ std::array<arg_t, OUTPUT_VEC_SIZE> accumulate_in_output(
      std::array<out_scalar_t*, OUTPUT_VEC_SIZE> out,
      std::array<arg_t, OUTPUT_VEC_SIZE> value,
      typename std::enable_if_t<can_acc>* = nullptr) const {
    std::array<arg_t, OUTPUT_VEC_SIZE> ret;
#pragma unroll
    for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
      kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
          &(ret[i]), out[i], ops, false);
      kps::Reduce<arg_t, 1, 1, ops_t, kps::details::ReduceMode::kLocalMode>(
          &(ret[i]), &(value[i]), ops, false);
    }
    return ret;
  }

  template <bool can_acc>
  __device__ out_scalar_t
  get_accumulated_output(out_scalar_t* out,
                         arg_t value,
                         typename std::enable_if_t<can_acc>* = nullptr) const {
    return (out_scalar_t)value;
  }

  template <int OUTPUT_VEC_SIZE, bool can_acc>
  __device__ std::array<arg_t, OUTPUT_VEC_SIZE> accumulate_in_output(
      std::array<out_scalar_t*, OUTPUT_VEC_SIZE>,
      std::array<arg_t, OUTPUT_VEC_SIZE>,
      typename std::enable_if_t<!can_acc>* = nullptr) const {
    return {arg_t{}};
  }

  template <bool can_acc>
  __device__ out_scalar_t
  get_accumulated_output(out_scalar_t* out,
                         arg_t value,
                         typename std::enable_if_t<!can_acc>* = nullptr) const {
    return *out;
  }

  template <class T>
  __device__ void set_results(const T x, const index_t base_offset) const {
    auto res = reinterpret_cast<out_scalar_t*>(reinterpret_cast<char*>(dst[0]) +
                                               base_offset);
    *res = x;
  }

  template <class T1, class T2>
  __device__ void set_results(const thrust::pair<T1, T2> x,
                              const index_t base_offset) const {
    if (noutputs >= 1) {
      auto res0 =
          reinterpret_cast<T1*>(reinterpret_cast<char*>(dst[0]) + base_offset);
      *res0 = x.first;
    }
  }

  template <int OUTPUT_VEC_SIZE>
  __device__ std::array<arg_t, OUTPUT_VEC_SIZE> global_reduce(
      std::array<arg_t, OUTPUT_VEC_SIZE> value, char* shared_memory) const {
    using arg_vec_t = std::array<arg_t, OUTPUT_VEC_SIZE>;
    using out_ptr_vec_t = std::array<out_scalar_t*, OUTPUT_VEC_SIZE>;
    using offset_vec_t = std::array<index_t, OUTPUT_VEC_SIZE>;

    arg_vec_t* reduce_buffer = reinterpret_cast<arg_vec_t*>(cta_buf);
    index_t output_idx = config.output_idx<OUTPUT_VEC_SIZE>();
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

#pragma unroll
    for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = reinterpret_cast<out_scalar_t*>(reinterpret_cast<char*>(dst[0]) +
                                               base_offsets[i]);
    }

    bool should_store = config.should_store(output_idx);
    if (should_store) {
      index_t offset = config.staging_memory_offset(blockIdx.y);
      reduce_buffer[offset] = value;
    }

    __threadfence();
    __syncthreads();
    bool is_last_block_done = mark_block_finished();

    if (is_last_block_done) {
      __threadfence();
      for (auto& v : value) {
        v = ident;
      }
      if (config.should_block_x_reduce()) {
        index_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
        index_t step = blockDim.x * blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          index_t idx = config.staging_memory_offset(input_offset);
          arg_vec_t next = reduce_buffer[idx];
#pragma unroll
          for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
            kps::Reduce<arg_t,
                        1,
                        1,
                        ops_t,
                        kps::details::ReduceMode::kLocalMode>(
                &(value[i]), &(next[i]), ops, false);
          }
        }
      } else {
        index_t input_offset = threadIdx.y;
        index_t step = blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          index_t idx = config.staging_memory_offset(input_offset);
          arg_vec_t next = reduce_buffer[idx];
#pragma unroll
          for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
            kps::Reduce<arg_t,
                        1,
                        1,
                        ops_t,
                        kps::details::ReduceMode::kLocalMode>(
                &(value[i]), &(next[i]), ops, false);
          }
        }
      }

      value = block_y_reduce<OUTPUT_VEC_SIZE>(value, shared_memory);

      if (config.should_block_x_reduce()) {
        value = block_x_reduce<OUTPUT_VEC_SIZE>(value, shared_memory);
      }

      if (should_store) {
#pragma unroll
        for (int i = 0; i < OUTPUT_VEC_SIZE; i++) {
          if (is_mean) {
            value[i] = value[i] / static_cast<arg_t>(mean_factor);
          }
          *(out[i]) = get_accumulated_output<can_accumulate_in_output>(
              out[i], value[i]);
        }
      }
    }

    return value;
  }
};

template <typename Context, int max_threads, typename R>
static void launch_reduce_kernel(const Context& dev_ctx,
                                 const ReduceStrideConfig& config,
                                 const R& reduction) {
  dim3 block = config.block();
  dim3 grid = config.grid();

  int shared_memory = config.shared_memory_size();
  auto stream = dev_ctx.stream();
  reduce_kernel<max_threads / 1, 1, R>
      <<<grid, block, shared_memory, stream>>>(reduction);
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

  constexpr int VT0 = 4;
  constexpr int INPUT_VEC_SIZE = 4;

  constexpr int base_idx = 0;

  ReduceStrideConfig reduce_config = setReduceConfig<T, T, VT0>(iter);

  void* buffer_data;
  void* semaphores_data;

  DenseTensor buffer_tensor;
  DenseTensor semaphore_tensor;

  std::vector<int> buffer_size = {static_cast<int>(
      reduce_config.global_memory_size() / phi::SizeOf(x.dtype()))};
  std::vector<int> semaphore_size = {static_cast<int>(
      reduce_config.semaphore_size() / phi::SizeOf(x.dtype()))};

  if (reduce_config.should_global_reduce()) {
    buffer_tensor.Resize(common::make_ddim(buffer_size));
    semaphore_tensor.Resize(common::make_ddim(semaphore_size));

    buffer_data =
        reinterpret_cast<void*>(dev_ctx.template Alloc<T>(&buffer_tensor));
    semaphores_data =
        reinterpret_cast<void*>(dev_ctx.template Alloc<T>(&semaphore_tensor));

    auto stream = dev_ctx.stream();
    phi::backends::gpu::GpuMemsetAsync(
        semaphores_data, 0, reduce_config.semaphore_size(), stream);
  }

  auto output_calc = make_output_calculator<uint32_t>(iter);
  auto input_calc = make_input_calculator<uint32_t>(iter);

  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  auto reducer = reduce_op<MPType>();

  int64_t mean_factor = iter.numel();

  auto reduce =
      ReduceStrideOp<T, reduce_op<MPType>, uint32_t, T, VT0, INPUT_VEC_SIZE>(
          reducer,
          reduce_config,
          input_calc,
          output_calc,
          in_data,
          out_data,
          buffer_data,
          reinterpret_cast<int*>(semaphores_data),
          ident,
          noutputs,
          base_idx,
          IsMean,
          mean_factor);

  reduce.accumulate = iter.should_accumulate();
  reduce.final_output = iter.is_final_output();

  constexpr int MAX_THREAD = 512;

  launch_reduce_kernel<Context, MAX_THREAD>(dev_ctx, reduce_config, reduce);
}

}  // namespace phi

#endif
