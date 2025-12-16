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

#pragma once

#include <bitset>
#include <limits>
#include <set>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
// #include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/legacy/reduce_max_kernel.h"
#include "paddle/phi/kernels/prod_kernel.h"
#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "paddle/phi/kernels/reduce_amin_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/reduce_min_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#ifndef PADDLE_WITH_XPU_KP
#include "paddle/phi/kernels/funcs/eigen/common.h"
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"

#include "paddle/phi/kernels/FunctionTraits.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"

#define WARP_SIZE 32

using DimMask = std::bitset<64>;

template <typename T>
struct LoadImpl {
  HOSTDEVICE static T apply(const void* src) {
    return *reinterpret_cast<const T*>(src);
  }
};

template <>
struct LoadImpl<bool> {
  HOSTDEVICE static bool apply(const void* src) {
    static_assert(sizeof(bool) == sizeof(char));
    // NOTE: [Loading boolean values]
    // Protect against invalid boolean values by loading as a byte
    // first, then converting to bool (see gh-54789).
    return *reinterpret_cast<const unsigned char*>(src);
  }
};

template <typename T>
HOSTDEVICE constexpr T load(const void* src) {
  return LoadImpl<T>::apply(src);
}

template <typename scalar_t>
HOSTDEVICE constexpr scalar_t load(const scalar_t* src) {
  return LoadImpl<scalar_t>::apply(src);
}

namespace phi {

constexpr size_t dim_bitset_size = 64;

inline std::bitset<dim_bitset_size> dim_list_to_bitset(
    std::vector<int> opt_dims, size_t ndims) {
  std::bitset<dim_bitset_size> seen;
  if (opt_dims.size() > 0) {
    for (int i = 0; i < opt_dims.size(); i++) {
      seen[opt_dims[i]] = true;
    }
  } else {
    for (size_t dim = 0; dim < ndims; dim++) {
      seen[dim] = true;
    }
  }
  return seen;
}

inline DenseTensor review_reduce_result(const DenseTensor& result,
                                        int ndim,
                                        DimMask mask,
                                        bool keep_dim = false) {
  if (keep_dim) {
    return result;
  }
  auto shape = common::vectorize(result.dims());
  auto stride = common::vectorize(result.strides());

  for (int dim = 0; dim < ndim; dim++) {
    if (mask[dim]) {
      shape.insert(shape.begin() + dim, 1);
      stride.insert(stride.begin() + dim, 0);
    }
  }

  return funcs::as_strided(result, shape, stride);
}

inline DimMask make_dim_mask(std::vector<int> opt_dims,
                             int64_t ndim,
                             bool allow_empty_dims = false) {
  DimMask mask;
  if (opt_dims.size() >= 0) {
    if (opt_dims.size() == 0 && !allow_empty_dims) {
      mask = DimMask().flip();
    } else {
      mask = dim_list_to_bitset(opt_dims, ndim);
    }
  } else {
    mask = DimMask().flip();
  }
  return mask;
}

template <typename T, int Size>
DEVICE AlignedVector<T, Size> load_vector(const T* base_ptr, uint32_t offset) {
  using vec_t = AlignedVector<T, Size>;
  auto* from = reinterpret_cast<const vec_t*>(base_ptr);
  return from[offset];
}

template <int Size>
DEVICE AlignedVector<bool, Size> load_vector(const bool* base_ptr,
                                             uint32_t offset) {
  // See NOTE [Loading boolean values]
  auto tmp = load_vector<uint8_t, Size>(
      reinterpret_cast<const uint8_t*>(base_ptr), offset);
  AlignedVector<bool, Size> ret;
  for (int i = 0; i < Size; ++i) {
    ret.val[i] = static_cast<bool>(tmp.val[i]);
  }
  return ret;
}

static inline int64_t div_up(int64_t a, int64_t b) { return (a + b - 1) / b; }

// returns floor(log2(n))
static inline int last_pow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

// template for changing MAX_NUM_THREADS based on op dtype
template <typename T>
struct mnt_wrapper {
  static constexpr int MAX_NUM_THREADS = 512;
};

template <int nt, int output_vec_size, typename R>
__launch_bounds__(nt, 4) __global__ void reduce_kernel(R reduction) {
  reduction.template run<output_vec_size>();
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

template <typename scalar_t>
int get_output_vec_size(const DenseTensorIterator& iter) {
  int vec_size = 4;
  auto update_vec_size = [&vec_size](uint64_t n) {
    while (n % vec_size != 0) {
      vec_size /= 2;
    }
  };

  uint64_t base_address =
      reinterpret_cast<uint64_t>(iter.data_ptr(iter.noutputs())) /
      sizeof(scalar_t);
  update_vec_size(base_address);

  const int output_index = iter.num_reduce_dims();
  update_vec_size(iter.shape()[output_index]);

  int j = 0;
  for (auto i : iter.strides(iter.noutputs())) {
    if (j != output_index) {
      update_vec_size(i / sizeof(scalar_t));
    }
    j++;
  }
  return vec_size;
}

HOSTDEVICE static void reduce_fraction(size_t* numerator, size_t* denominator) {
  size_t a = *denominator;
  size_t b = *numerator;
  while (b != 0) {
    a %= b;
    size_t tmp = a;
    a = b;
    b = tmp;
  }

  *numerator /= a;
  *denominator /= a;
}

struct ReduceConfig {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
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

  template <typename T>
  void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_num_threads =
        mnt_wrapper<T>::MAX_NUM_THREADS / output_vec_size;
    int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(last_pow2(dim0))
                                           : max_num_threads;
    int dim1_pow2 = dim1 < max_num_threads ? static_cast<int>(last_pow2(dim1))
                                           : max_num_threads;
    block_width = std::min(dim0_pow2, WARP_SIZE);
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
    return dim3(div_up(num_outputs / output_vec_size, step_output),
                ctas_per_output);
  }

  HOSTDEVICE bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  HOSTDEVICE bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  HOSTDEVICE bool should_global_reduce() const { return input_mult[CTA] != 0; }

  DEVICE bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
           (!should_block_x_reduce() || threadIdx.x == 0) &&
           (!should_block_y_reduce() || threadIdx.y == 0);
  }

  DEVICE bool should_reduce_tail() const {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
           (!should_global_reduce() || blockIdx.y == 0);
  }

  HOSTDEVICE int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] + warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  template <int output_vec_size>
  HOSTDEVICE int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] + warp * output_mult[BLOCK_Y] +
            cta1 * step_output) *
           output_vec_size;
  }

  DEVICE int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  DEVICE int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() || block_width <= WARP_SIZE)) {
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

  int values_per_thread() const { return div_up(num_inputs, step_input); }

  int mock_values_per_thread(int parallelism) {
    return div_up(num_inputs, step_input * parallelism);
  }
};

template <typename arg_t, typename scalar_t, int kVt0, int kInputVecSize = kVt0>
ReduceConfig setReduceConfig(const DenseTensorIterator& iter) {
  int device_id = paddle::platform::GetCurrentDeviceId();

  // Start by assuming that each thread handles a single output and all
  // the inputs for that output.
  int64_t num_outputs = iter.num_output_elements();
  // num_outputs=5;
  int64_t inputs_per_output = iter.numel() / num_outputs;
  int input_index = iter.ntensors() - 1;

  auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output);

  int64_t dim0;
  int64_t dim1;
  int64_t fastest_moving_stride;
  bool reduction_on_fastest_striding_dimension;

  if (iter.ndim() > 0) {
    // Adjust block size to map block width to fastest changing dimension of
    // input tensor. This grants the best possible memory accessing pattern,
    // given that for non-contiguous tensor with space in between, we cannot
    // have perfect memory coalescing.
    reduction_on_fastest_striding_dimension =
        (iter.num_reduce_dims() == iter.ndim()) ||
        (iter.strides(/*arg=*/input_index)[0] <
         iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()]);

    // Notice that dim0 & dim1 does NOT guarantee any launch configuration here!
    // dim0 & dim1 are more like the upper bound of the block dimension. The
    // actual launch config and reduction scheme is determined by setting values
    // to `config.input_mult` and `config.output_mult`.
    // We try to max out dim1 so that we have enough threads per CTA to deliver
    // performance for larger problem size.
    if (reduction_on_fastest_striding_dimension) {
      // Map block.x to the fastest reducing dimension. It implies:
      //   1. block_x_reduce is required.
      //   2. block.y now max out to num_outputs.
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = iter.strides(/*arg=*/input_index)[0];
    } else {
      // Map block.x to the fastest non reducing dimension. It implies:
      //   1. block_x_reduce is turned off.
      //   2. block.y now max out to inputs_per_output.
      dim0 = num_outputs;
      dim1 = inputs_per_output;
      fastest_moving_stride =
          iter.strides(/*arg=*/input_index)[iter.num_reduce_dims()];
    }
  } else {
    reduction_on_fastest_striding_dimension = true;
    fastest_moving_stride = sizeof(scalar_t);
    dim0 = 1;
    dim1 = 1;
  }

  // We do vectorization to gain better memory access, there are two cases which
  // we call "vectorize along input" and "vectorize along output". Note that the
  // "input/output" here does not mean we are vectorizing load/store
  // instructions. We always only vectorize load instructions.
  //
  // Case 1: "vectorize along input"
  // This case happens when we are reducing along fastest moving dimension. In
  // such case, threads with the same threadIdx.y works on the same reduction
  // cooperatively and will produce results for the same output. In such case,
  // values in each loaded vector always correspond to the same output.
  //
  // Case 2: "vectorize along output"
  // This case happens when the fastest moving dimension is not the dimension of
  // reduction. In such case, threads with different threadIdx.x are independent
  // and will produce results for different outputs. In such case, values in
  // each loaded vector always correspond to different outputs.
  if (fastest_moving_stride == sizeof(scalar_t)) {
    if (reduction_on_fastest_striding_dimension && dim0 > 128 &&
        iter.num_reduce_dims() == 1 && kVt0 >= kInputVecSize) {
      // Case 1: "vectorize along input"
      // Note that if kVt0 < ReduceConfig::vec_size, then this means the
      // register pressure could be high, in such case, we should avoid
      // vectorization.
      config.vectorize_input = true;
      dim0 /= kInputVecSize;
    } else if (!reduction_on_fastest_striding_dimension) {
      // Case 2: "vectorize along output"
      config.output_vec_size = get_output_vec_size<scalar_t>(iter);
      dim0 /= config.output_vec_size;
    }
  }

  // Adjust block_width and block_height
  config.set_block_dimension<scalar_t>(dim0, dim1);

  int block_width = config.block_width;
  int block_height = config.block_height;

  if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
    // Split the input across lanes if the input is contiguous in the reduced
    // dimension. This will require reduction between threads using warp
    // shuffle instructions and shared memory (if block_width > WARP_SIZE).
    config.input_mult[0] = config.split_input(block_width);
  } else {
    // Otherwise split the output across lanes in a warp.
    config.output_mult[0] = config.split_output(block_width);
  }

  constexpr int min_values_per_thread = 16;
  constexpr int max_values_per_thread = 256;

  const int warp_split_threshold =
      std::min<int>(block_height * 16, max_values_per_thread);
  bool split_across_warps = config.values_per_thread() >= warp_split_threshold;

  const int num_mp = paddle::platform::GetGPUMultiProcessors(device_id);

  if (split_across_warps) {
    // Divide the input across warps in a thread-block, if that leaves at least
    // 16 elements to be summed by each thread. This will require inter-warp
    // reduction using shared memory.
    config.input_mult[1] = config.split_input(block_height);
  } else {
    // Otherwise, each warp handles a separate output.
    config.output_mult[1] = config.split_output(block_height);
  }

  int max_threads_per_mp =
      paddle::platform::GetGPUMaxThreadsPerMultiProcessor(device_id);

  const int blocks_per_sm = max_threads_per_mp / config.num_threads;
  const int target_grid_size = num_mp * blocks_per_sm;
  int grid = config.grid().x;
  if (config.input_mult[1] != 0 &&
      config.values_per_thread() >= max_values_per_thread &&
      grid <= target_grid_size) {
    // Divide the input across thread-blocks if the amount of work per-thread
    // is large enough and the size of the output is small enough. This will
    // require a reduction using global memory.
    // If we decide to split input across blocks, as long as we can get enough
    // number of blocks (`target_grid_size`) to balance SM, we should still
    // make the number of values per thread large for best performance.
    int ctas_per_output1 = div_up(target_grid_size, grid);
    int ctas_per_output2 =
        div_up(config.values_per_thread(), min_values_per_thread);
    int ctas_per_output3 =
        div_up(config.values_per_thread(), max_values_per_thread);
    // We want the minimum of ctas_per_output1 and ctas_per_output2, so that
    // each thread can have a large number of values to deal with. But we don't
    // want values_per_thread to be larger than max_values_per_thread
    config.ctas_per_output = std::max(
        std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);

    if (config.ctas_per_output > 1) {
      config.input_mult[2] = config.split_input(config.ctas_per_output);
    }
  }
  return config;
}

template <typename scalar_t,
          typename ops_t,
          typename index_t,
          typename out_scalar_t = scalar_t,
          int kVt0 = 4,
          int kInputVecSize = kVt0>
struct ReduceOp {
  using traits = function_traits<decltype(&ops_t::reduce)>;
  using arg_t =
      typename std::decay<typename traits::template arg<0>::type>::type;

  using InputCalculator = funcs::OffsetCalculator<1, index_t>;
  using OutputCalculator = funcs::OffsetCalculator<2, index_t>;

  static constexpr bool can_accumulate_in_output =
      std::is_convertible_v<arg_t, out_scalar_t> &&
      std::is_convertible_v<out_scalar_t, arg_t>;

  ops_t ops;
  arg_t ident;
  ReduceConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void* src;
  char* dst[2];  // it accepts at most two destinations
  // acc_buf used for accumulation among sub Tensor Iterator when accumulation
  // on output is not permissible
  void* acc_buf;
  // cta_buf used for accumulation between blocks during global reduction
  void* cta_buf;
  int* semaphores;
  int64_t base_idx;
  bool accumulate;
  bool final_output;
  int noutputs;

  ReduceOp(ops_t ops,
           ReduceConfig config,
           InputCalculator input_calc,
           OutputCalculator output_calc,
           const void* src,
           char* dst0,
           std::optional<char*> dst1,
           void* acc_buf,
           void* cta_buf,
           int* semaphores,
           arg_t ident,
           int noutputs,
           int64_t base_idx)
      : ops(ops),
        ident(ident),
        config(config),
        input_calc(input_calc),
        output_calc(output_calc),
        src(src),
        acc_buf(acc_buf),
        cta_buf(cta_buf),
        semaphores(semaphores),
        base_idx(base_idx),
        noutputs(noutputs) {
    dst[0] = dst0;
    if (dst1.has_value()) {
      dst[1] = dst1.value();
    }
  }

  template <int output_vec_size>
  DEVICE void run() const {
    extern __shared__ char shared_memory[];
    index_t output_idx = config.output_idx<output_vec_size>();
    index_t input_idx = config.input_idx();
    auto base_offsets1 = output_calc.get(output_idx)[1];

    using arg_vec_t = std::array<arg_t, output_vec_size>;
    arg_vec_t value;

    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      const scalar_t* input_slice =
          (const scalar_t*)((const char*)src + base_offsets1);
      value = thread_reduce<output_vec_size>(input_slice);
    }

    if (config.should_block_y_reduce()) {
      value = block_y_reduce<output_vec_size>(value, shared_memory);
    }

    if (config.should_block_x_reduce()) {
      value = block_x_reduce<output_vec_size>(value, shared_memory);
    }

    using out_ptr_vec_t = std::array<out_scalar_t*, output_vec_size>;
    using offset_vec_t = std::array<index_t, output_vec_size>;
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

#pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = reinterpret_cast<out_scalar_t*>(dst[0] + base_offsets[i]);
    }

    arg_vec_t* acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(arg_t);
      size_t denominator = sizeof(out_scalar_t);
      reduce_fraction(&numerator, &denominator);
      acc = reinterpret_cast<arg_vec_t*>(
          reinterpret_cast<char*>(acc_buf) +
          (base_offsets[0] * numerator / denominator));
    }

    if (config.should_global_reduce()) {
      value = global_reduce<output_vec_size>(value, acc, shared_memory);
    } else if (config.should_store(output_idx)) {
      if (accumulate) {
#pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = ops.translate_idx(value[i], base_idx);
        }
      }

      if (acc == nullptr) {
        if (accumulate) {
          value =
              accumulate_in_output<output_vec_size, can_accumulate_in_output>(
                  out, value);
        }
        if (final_output) {
          set_results_to_output<output_vec_size>(value, base_offsets);
        } else {
#pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            *(out[i]) = get_accumulated_output<can_accumulate_in_output>(
                out[i], value[i]);
          }
        }
      } else {
        if (accumulate) {
#pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = ops.combine((*acc)[i], value[i]);
            // value[i] = reducer((*acc)[i], value[i]);
          }
        }
        if (final_output) {
          set_results_to_output<output_vec_size>(value, base_offsets);
        } else {
          *acc = value;
        }
      }
    }
  }

  template <int output_vec_size>
  DEVICE std::array<arg_t, output_vec_size> thread_reduce(
      const scalar_t* data) const {
    if (config.vectorize_input) {
      // CUDA_KERNEL_ASSERT(output_vec_size == 1);
      // reduce at the header of input_slice where memory is not aligned,
      // so that thread_reduce will have an aligned memory to work on.
      return {input_vectorized_thread_reduce_impl(data)};
    } else {
      index_t element_stride = input_calc.strides_[0][0] / sizeof(scalar_t);
      bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
      if (is_contiguous) {
        return thread_reduce_impl<output_vec_size>(
            data, [](index_t idx) { return idx; });
      } else if (input_calc.dims == 1) {
        return thread_reduce_impl<output_vec_size>(
            data, [&](index_t idx) { return idx * element_stride; });
      } else {
        return thread_reduce_impl<output_vec_size>(data, [&](index_t idx) {
          return input_calc.get(idx)[0] / sizeof(scalar_t);
        });
      }
    }
  }

  DEVICE arg_t input_vectorized_thread_reduce_impl(const scalar_t* data) const {
    index_t end = config.num_inputs;

    // Handle the head of input slice where data is not aligned
    arg_t value = ident;

    // constexpr int align_bytes =
    // alignof(at::native::memory::aligned_vector<scalar_t, kInputVecSize>);
    constexpr int align_bytes =
        alignof(phi::AlignedVector<scalar_t, kInputVecSize>);

    constexpr int align_elements = align_bytes / sizeof(scalar_t);
    int shift = ((uint64_t)data) % align_bytes / sizeof(scalar_t);

    if (shift > 0) {
      data -= shift;
      end += shift;
      if (threadIdx.x >= shift && threadIdx.x < align_elements &&
          config.should_reduce_tail()) {
        value =
            ops.reduce(value, load(data + threadIdx.x), threadIdx.x - shift);
        // value = reducer(value, load(data + threadIdx.x));
      }
      end -= align_elements;
      data += align_elements;
      shift = align_elements - shift;
    }

    index_t idx = config.input_idx();
    const index_t stride = config.step_input;

    // Multiple accumulators to remove dependency between unrolled loops.
    arg_t value_list[kInputVecSize];
    value_list[0] = value;

#pragma unroll
    for (int i = 1; i < kInputVecSize; i++) {
      value_list[i] = ident;
    }

    // Do the vectorized reduction
    using load_t = phi::AlignedVector<scalar_t, kInputVecSize>;

    while (idx * kInputVecSize + kInputVecSize - 1 < end) {
      const auto values_vec = load_vector<scalar_t, kInputVecSize>(data, idx);

#pragma unroll
      for (index_t i = 0; i < kInputVecSize; i++) {
        value_list[i] = ops.reduce(
            value_list[i], values_vec.val[i], shift + idx * kInputVecSize + i);
        // value_list[i] = reducer(value_list[i], values_vec.val[i]);
      }
      idx += stride;
    }

    // tail
    index_t tail_start = end - end % kInputVecSize;

    if (config.should_reduce_tail()) {
      int idx = tail_start + threadIdx.x;
      if (idx < end) {
        const auto value = load(data + idx);
        value_list[0] = ops.reduce(value_list[0], value, idx + shift);
        // value_list[0] = reducer(value_list[0], value);
      }
    }

// combine accumulators
#pragma unroll
    for (int i = 1; i < kInputVecSize; i++) {
      value_list[0] = ops.combine(value_list[0], value_list[i]);
      // value_list[0] = reducer(value_list[0], value_list[i]);
    }
    return value_list[0];
  }

  template <int output_vec_size, typename offset_calc_t>
  DEVICE std::array<arg_t, output_vec_size> thread_reduce_impl(
      const scalar_t* data_, offset_calc_t calc) const {
    index_t idx = config.input_idx();
    const index_t end = config.num_inputs;
    const index_t stride = config.step_input;

    using arg_vec_t = std::array<arg_t, output_vec_size>;
    using load_t = phi::AlignedVector<scalar_t, output_vec_size>;

    // Multiple accumulators to remove dependency between unrolled loops.
    arg_vec_t value_list[kVt0];

#pragma unroll
    for (int i = 0; i < kVt0; i++) {
#pragma unroll
      for (int j = 0; j < output_vec_size; j++) {
        value_list[i][j] = ident;
      }
    }

    load_t values[kVt0];

    while (idx + (kVt0 - 1) * stride < end) {
#pragma unroll
      for (index_t i = 0; i < kVt0; i++) {
        const auto offset = calc(idx + i * stride) / output_vec_size;

        // values[i] = memory::load_vector<output_vec_size>(data_, offset);
        // phi::MyLoad<scalar_t, output_vec_size>(&data_[offset], &values[i]);
        values[i] = load_vector<scalar_t, output_vec_size>(data_, offset);
      }
#pragma unroll
      for (index_t i = 0; i < kVt0; i++) {
#pragma unroll
        for (index_t j = 0; j < output_vec_size; j++) {
          value_list[i][j] =
              ops.reduce(value_list[i][j], values[i].val[j], idx + i * stride);
          // value_list[i][j] = reducer(value_list[i][j], values[i].val[j]);
        }
      }
      idx += stride * kVt0;
    }

    // tail
    int idx_ = idx;
#pragma unroll
    for (index_t i = 0; i < kVt0; i++) {
      if (idx >= end) {
        break;
      }
      const auto offset = calc(idx) / output_vec_size;
      values[i] = load_vector<scalar_t, output_vec_size>(data_, offset);
      // phi::MyLoad<scalar_t, output_vec_size>(&data_[offset], &values[i]);
      idx += stride;
    }
    idx = idx_;
#pragma unroll
    for (index_t i = 0; i < kVt0; i++) {
      if (idx >= end) {
        break;
      }
#pragma unroll
      for (index_t j = 0; j < output_vec_size; j++) {
        value_list[i][j] = ops.reduce(value_list[i][j], values[i].val[j], idx);
        // value_list[i][j] = reducer(value_list[i][j], values[i].val[j]);
      }
      idx += stride;
    }

// combine accumulators
#pragma unroll
    for (int i = 1; i < kVt0; i++) {
#pragma unroll
      for (index_t j = 0; j < output_vec_size; j++) {
        value_list[0][j] = ops.combine(value_list[0][j], value_list[i][j]);
        // value_list[0][j] = reducer(value_list[0][j], value_list[i][j]);
      }
    }
    return value_list[0];
  }

  template <int output_vec_size>
  DEVICE std::array<arg_t, output_vec_size> block_x_reduce(
      std::array<arg_t, output_vec_size> value, char* shared_memory) const {
    using args_vec_t = std::array<arg_t, output_vec_size>;
    int dim_x = blockDim.x;
    args_vec_t* shared = reinterpret_cast<args_vec_t*>(shared_memory);
    if (dim_x > WARP_SIZE) {
      int address_base = threadIdx.x + threadIdx.y * blockDim.x;

      shared[address_base] = value;
      for (int offset = dim_x / 2; offset >= WARP_SIZE; offset >>= 1) {
        __syncthreads();

        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
          args_vec_t other = shared[address_base + offset];

#pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = ops.combine(value[i], other[i]);
            // value[i] = reducer(value[i], other[i]);
          }
          shared[address_base] = value;
        }
      }
      dim_x = WARP_SIZE;
    }

    __syncthreads();

    // Intra-warp reduction, fix CUDA to have offset decreasing for better
    // numerics matching Triton, etc.
    // TODO(PaulZhang12): AMD and internal

    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);
    // for (int offset = dim_x >> 1; offset > 0; offset >>= 1) {
    for (int offset = 1; offset < dim_x; offset <<= 1) {
#pragma unroll
      for (int i = 0; i < output_vec_size; i++) {
        arg_t other = ops.warp_shfl_down(value[i], offset);
        value[i] = ops.combine(value[i], other);
        // arg_t other = phi::backends::gpu::CudaShuffleDownSync(mask, value[i],
        // offset); value[i] = reducer(value[i], other);
      }
    }
    return value;
  }

  template <int output_vec_size>
  DEVICE std::array<arg_t, output_vec_size> block_y_reduce(
      std::array<arg_t, output_vec_size> value, char* shared_memory) const {
    using args_vec_t = std::array<arg_t, output_vec_size>;
    args_vec_t* shared = reinterpret_cast<args_vec_t*>(shared_memory);
    shared[config.shared_memory_offset(0)] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
        args_vec_t other = shared[config.shared_memory_offset(offset)];
#pragma unroll
        for (int i = 0; i < output_vec_size; i++) {
          value[i] = ops.combine(value[i], other[i]);
          // value[i] = reducer(value[i], other[i]);
        }
        shared[config.shared_memory_offset(0)] = value;
      }
    }

    return value;
  }

  DEVICE bool mark_block_finished() const {
    __shared__ bool is_last_block_done_shared;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
    }

    __syncthreads();

    return is_last_block_done_shared;
  }

  template <int output_vec_size, bool can_acc>
  DEVICE std::array<arg_t, output_vec_size> accumulate_in_output(
      std::array<out_scalar_t*, output_vec_size> out,
      std::array<arg_t, output_vec_size> value,
      typename std::enable_if_t<can_acc>* = nullptr) const {
    std::array<arg_t, output_vec_size> ret;
#pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      ret[i] = ops.combine(*(out[i]), value[i]);
      // ret[i] = reducer(*(out[i]), value[i]);
    }
    return ret;
  }

  template <bool can_acc>
  DEVICE out_scalar_t
  get_accumulated_output(out_scalar_t* out,
                         arg_t value,
                         typename std::enable_if_t<can_acc>* = nullptr) const {
    // CUDA_KERNEL_ASSERT(!final_output);
    return (out_scalar_t)value;
  }

  // This function should never be called --
  // it's the version of `accumulate_in_output`
  // when accumulation in the output is not possible.
  template <int output_vec_size, bool can_acc>
  DEVICE std::array<arg_t, output_vec_size> accumulate_in_output(
      std::array<out_scalar_t*, output_vec_size>,
      std::array<arg_t, output_vec_size>,
      typename std::enable_if_t<!can_acc>* = nullptr) const {
    // CUDA_KERNEL_ASSERT(false);
    return {arg_t{}};
  }

  // This function should never be called --
  // it's the version of `get_accumulated_output`
  // when accumulation in the output is not possible.
  template <bool can_acc>
  DEVICE out_scalar_t
  get_accumulated_output(out_scalar_t* out,
                         arg_t value,
                         typename std::enable_if_t<!can_acc>* = nullptr) const {
    // CUDA_KERNEL_ASSERT(false);
    return *out;
  }

  template <class T>
  DEVICE void set_results(const T x, const index_t base_offset) const {
    // CUDA_KERNEL_ASSERT(noutputs == 1);
    auto res = reinterpret_cast<out_scalar_t*>(dst[0] + base_offset);
    *res = x;
  }

  // Currently implemented for max of two outputs
  template <class T1, class T2>
  DEVICE void set_results(const thrust::pair<T1, T2> x,
                          const index_t base_offset) const {
    if (noutputs >= 1) {
      auto res0 = reinterpret_cast<T1*>(dst[0] + base_offset);
      *res0 = x.first;
    }
    if (noutputs >= 2) {
      // base offset is computed assuming element size being sizeof(T1), so we
      // need to make a correction to obtain the correct base offset
      auto res1 =
          reinterpret_cast<T2*>(dst[1] + base_offset / sizeof(T1) * sizeof(T2));
      *res1 = x.second;
    }
  }

  template <int output_vec_size>
  DEVICE void set_results_to_output(
      std::array<arg_t, output_vec_size> value,
      std::array<index_t, output_vec_size> base_offset) const {
// CUDA_KERNEL_ASSERT(final_output);
#pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      set_results(ops.project(value[i]), base_offset[i]);
      // set_results(static_cast<out_scalar_t>(value[i]), base_offset[i]);
    }
  }

  template <int output_vec_size>
  DEVICE std::array<arg_t, output_vec_size> global_reduce(
      std::array<arg_t, output_vec_size> value,
      std::array<arg_t, output_vec_size>* acc,
      char* shared_memory) const {
    using arg_vec_t = std::array<arg_t, output_vec_size>;
    using out_ptr_vec_t = std::array<out_scalar_t*, output_vec_size>;
    using offset_vec_t = std::array<index_t, output_vec_size>;

    arg_vec_t* reduce_buffer = reinterpret_cast<arg_vec_t*>(cta_buf);
    index_t output_idx = config.output_idx<output_vec_size>();
    offset_vec_t base_offsets;
    out_ptr_vec_t out;

#pragma unroll
    for (int i = 0; i < output_vec_size; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = reinterpret_cast<out_scalar_t*>(dst[0] + base_offsets[i]);
    }

    bool should_store = config.should_store(output_idx);
    if (should_store) {
      index_t offset = config.staging_memory_offset(blockIdx.y);
      reduce_buffer[offset] = value;
    }

    __threadfence();  // make sure writes are globally visible

    __syncthreads();  // if multiple warps in this block wrote to staging, make
                      // sure they're all done
    bool is_last_block_done = mark_block_finished();

    if (is_last_block_done) {
      __threadfence();  // complete the acquire pattern after atomic

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
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = ops.combine(value[i], next[i]);
            // value[i] = reducer(value[i], next[i]);
          }
        }
      } else {
        index_t input_offset = threadIdx.y;
        index_t step = blockDim.y;

        for (; input_offset < config.ctas_per_output; input_offset += step) {
          index_t idx = config.staging_memory_offset(input_offset);
          arg_vec_t next = reduce_buffer[idx];
#pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = ops.combine(value[i], next[i]);
            // value[i] = reducer(value[i], next[i]);
          }
        }
      }
      value = block_y_reduce<output_vec_size>(value, shared_memory);
      if (config.should_block_x_reduce()) {
        value = block_x_reduce<output_vec_size>(value, shared_memory);
      }
      if (should_store) {
        if (accumulate) {
#pragma unroll
          for (int i = 0; i < output_vec_size; i++) {
            value[i] = ops.translate_idx(value[i], base_idx);
          }
        }

        if (acc == nullptr) {
          if (accumulate) {
            value =
                accumulate_in_output<output_vec_size, can_accumulate_in_output>(
                    out, value);
          }
          if (final_output) {
            set_results_to_output<output_vec_size>(value, base_offsets);
          } else {
#pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
              *(out[i]) = get_accumulated_output<can_accumulate_in_output>(
                  out[i], value[i]);
            }
          }
        } else {
          if (accumulate) {
#pragma unroll
            for (int i = 0; i < output_vec_size; i++) {
              value[i] = ops.combine((*acc)[i], value[i]);
              // value[i] = reducer((*acc)[i], value[i]);
            }
          }
          if (final_output) {
            set_results_to_output<output_vec_size>(value, base_offsets);
          } else {
            *acc = value;
          }
        }
      }
    }

    return value;
  }
};

class AccumulationBuffer {
 public:
  AccumulationBuffer() {}

  AccumulationBuffer(const KPDevice& dev_ctx,
                     size_t acc_t_size,
                     size_t out_t_size,
                     char* out_ptr,
                     int64_t size) {
    out_ptr_ = reinterpret_cast<char*>(out_ptr);
    if (out_t_size >= acc_t_size) {
      // reusing output buffer for accumulation.
      acc_ptr_ = reinterpret_cast<char*>(out_ptr);
      numerator_ = 1;
      denominator_ = 1;
    } else {
      // auto& allocator = *c10::cuda::CUDACachingAllocator::get();
      phi::Allocator* allocator =
          const_cast<phi::Allocator*>(&(dev_ctx.GetAllocator()));
      buffer_ = allocator->Allocate(size);
      acc_ptr_ = reinterpret_cast<char*>(buffer_->ptr()());
      numerator_ = acc_t_size;
      denominator_ = out_t_size;
      reduce_fraction(&numerator_, &denominator_);
    }
  }

  char* GetAccSlice(char* out_ptr) {
    if (acc_ptr_ == nullptr) {
      return nullptr;
    }
    return acc_ptr_ + ((out_ptr - out_ptr_) * numerator_ / denominator_);
  }

 private:
  char* acc_ptr_ = nullptr;
  char* out_ptr_ = nullptr;
  size_t numerator_;
  size_t denominator_;
  Allocator::AllocationPtr buffer_;
};

template <int max_threads, typename R>
static void launch_reduce_kernel(const ReduceConfig& config,
                                 const R& reduction,
                                 cudaStream_t stream) {
  dim3 block = config.block();
  dim3 grid = config.grid();

  int shared_memory = config.shared_memory_size();

  switch (config.output_vec_size) {
    case 4:
      reduce_kernel<max_threads / 4, 4, R>
          <<<grid, block, shared_memory, stream>>>(reduction);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
      break;
    case 2:
      reduce_kernel<max_threads / 2, 2, R>
          <<<grid, block, shared_memory, stream>>>(reduction);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
      break;
    default:
      reduce_kernel<max_threads / 1, 1, R>
          <<<grid, block, shared_memory, stream>>>(reduction);
      PADDLE_ENFORCE_GPU_SUCCESS(cudaGetLastError());
  }
}

template <typename Tx,
          typename Ty,
          int kVt0 = 4,
          int kInputVecSize = kVt0,
          typename ops_t,
          typename ident_t = double>
inline void gpu_reduce_kernel(const KPDevice& dev_ctx,
                              const DenseTensorIterator& iter,
                              const ops_t& ops,
                              ident_t ident = 0,
                              AccumulationBuffer* acc_buf_ptr = nullptr,
                              int64_t base_idx = 0) {
  auto stream = dev_ctx.stream();

  //   using arg_t = typename phi::dtype::MPTypeTrait<Ty>::Type;

  using traits = function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename traits::template arg<0>::type;

  // at::Half/at::ComplexHalf overflows easily as it's range is very small.
  // So when scalar_t and out_scalar_t are at::Half/at::ComplexHalf, we
  // set can_accumulate_in_output to False.
  static constexpr bool is_inp_out_type_half_or_chalf =
      (std::is_same_v<phi::float16, Tx> && std::is_same_v<phi::float16, Ty>) ||
      (std::is_same_v<phi::dtype::complex<float16>, Tx> &&
       std::is_same_v<phi::dtype::complex<float16>, Ty>);
  // at::BFloat16 has lower precision and can lead to rounding errors.
  // So when scalar_t and out_scalar_t are at::BFloat16, we
  // set can_accumulate_in_output to False.
  static constexpr bool is_inp_out_type_bfloat16 =
      (std::is_same_v<phi::bfloat16, Tx> && std::is_same_v<phi::bfloat16, Ty>);
  static constexpr bool can_accumulate_in_output =
      std::is_convertible_v<arg_t, Ty> &&
      !(is_inp_out_type_half_or_chalf || is_inp_out_type_bfloat16);

  bool can_use_32bit_indexing = iter.can_use_32bit_indexing();
  std::unique_ptr<AccumulationBuffer> owned_buf_ptr;
  // The acc_buf_ptr is a shared pointer. It is create at the first entrance and
  // reused by all recursive function calls.
  if (acc_buf_ptr == NULL) {
    // acc_buf_ptr holds buffer used for accumulation among multiple sub_iter
    // when accumulation in output is not possible.
    if (!can_accumulate_in_output && !can_use_32bit_indexing) {
      int64_t output_memory_size = sizeof(iter.dtype(0));
      for (int dim = 0; dim < iter.ndim(); dim++) {
        output_memory_size = std::max(output_memory_size,
                                      iter.shape()[dim] * iter.strides(0)[dim]);
      }
      output_memory_size /= sizeof(iter.dtype(0));
      owned_buf_ptr.reset(
          new AccumulationBuffer(dev_ctx,
                                 sizeof(arg_t),
                                 sizeof(Ty),
                                 reinterpret_cast<char*>(iter.data_ptr(0)),
                                 output_memory_size * sizeof(arg_t)));
    } else {
      owned_buf_ptr.reset(new AccumulationBuffer());
    }
    acc_buf_ptr = owned_buf_ptr.get();
  }

  if (!can_use_32bit_indexing) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      int64_t sub_iter_base_idx = sub_iter.view_offsets()[0];
      gpu_reduce_kernel<Tx, Ty, kVt0, kInputVecSize>(
          dev_ctx, sub_iter, ops, ident, acc_buf_ptr, sub_iter_base_idx);
    }
    return;
  }

  const char* in_data =
      reinterpret_cast<char*>(iter.data_ptr(iter.ntensors() - 1));
  char* out_data = reinterpret_cast<char*>(iter.data_ptr(0));
  const auto noutputs = iter.noutputs();

  char* acc_data = acc_buf_ptr->GetAccSlice(out_data);

  ReduceConfig config = setReduceConfig<arg_t, Tx, kVt0, kInputVecSize>(iter);

  Allocator::AllocationPtr buffer;
  Allocator::AllocationPtr semaphores;
  void* buffer_ptr;
  void* semaphores_ptr;

  if (config.should_global_reduce()) {
    phi::Allocator* allocator =
        const_cast<phi::Allocator*>(&(dev_ctx.GetAllocator()));
    buffer = allocator->Allocate(config.global_memory_size());
    semaphores = allocator->Allocate(config.semaphore_size());
    buffer_ptr = buffer->ptr();
    semaphores_ptr = semaphores->ptr();

    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemsetAsync(semaphores_ptr, 0, config.semaphore_size(), stream));
  }

  // AT_ASSERT(can_use_32bit_indexing);
  auto output_calc = make_output_calculator<uint32_t>(iter);
  auto input_calc = make_input_calculator<uint32_t>(iter);

  auto reduce = ReduceOp<Tx, ops_t, uint32_t, Ty, kVt0, kInputVecSize>(
      ops,
      config,
      input_calc,
      output_calc,
      in_data,
      out_data,
      nullptr,  // out_data_extra,
      acc_data,
      buffer_ptr,
      reinterpret_cast<int*>(semaphores_ptr),
      ident,
      noutputs,
      base_idx);

  reduce.accumulate = iter.should_accumulate();
  reduce.final_output = iter.is_final_output();

  launch_reduce_kernel<mnt_wrapper<Tx>::MAX_NUM_THREADS>(
      config, reduce, stream);

  return;
}

template <typename Tx,
          typename Ty,
          template <typename>
          class ops_t,
          typename TransformOp,
          bool IsMean = false>
void ReduceGpuKernel(const KPDevice& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* y,
                     const TransformOp& transform,
                     const std::vector<int>& origin_reduce_dims) {
  if (x.numel() == 0) {
    dev_ctx.Alloc<Ty>(y);
    return;
  }

  dev_ctx.Alloc<Ty>(y);

  int64_t ndim = x.dims().size();
  auto mask = make_dim_mask(origin_reduce_dims, ndim);
  auto viewed_result = review_reduce_result(*(y), ndim, mask);
  auto x_dim = common::vectorize<int64_t>(x.dims());

  if (x_dim.size() == 0) {
    std::vector<const DenseTensor*> inputs = {&x};
    // std::vector<DenseTensor*> outputs = {y};
    std::vector<DenseTensor*> outputs = {&viewed_result};
    funcs::ElementwiseKernel<Ty>(dev_ctx, inputs, &outputs, transform);
    return;
  }

  DenseTensorIteratorConfig dense_iter_config;
  dense_iter_config.is_reduction(true);
  dense_iter_config.add_output(viewed_result);
  dense_iter_config.add_const_input(x);
  DenseTensorIterator iter = dense_iter_config.build();
  constexpr int kVt0 = 4;
  constexpr int kInputVecSize = kVt0;
  using arg_t = typename phi::dtype::MPTypeTrait<Ty>::Type;
  auto reducer = ops_t<arg_t>();

  gpu_reduce_kernel<Tx, Ty, kVt0, kInputVecSize, ops_t<arg_t>>(
      dev_ctx, iter, reducer);
  return;
}
}  // namespace phi
