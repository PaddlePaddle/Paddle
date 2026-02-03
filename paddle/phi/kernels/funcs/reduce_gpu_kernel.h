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

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/legacy/reduce_max_kernel.h"
#include "paddle/phi/kernels/prod_kernel.h"
#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "paddle/phi/kernels/reduce_amin_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/reduce_min_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/phi/kernels/funcs/function_traits.h"
#include "paddle/phi/kernels/primitive/reduce_primitives.h"

#ifdef PADDLE_WITH_HIP
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

// The GPUReduceScheduler splits tensors with indices exceeding 32-bit range to
// ensure that all incoming tensors can be addressed within 32-bit index space.
using IndexType = uint32_t;

template <typename T>
struct LoadImpl {
  HOSTDEVICE static T Apply(const void* src) {
    return *reinterpret_cast<const T*>(src);
  }
};

template <>
struct LoadImpl<bool> {
  HOSTDEVICE static bool Apply(const void* src) {
    static_assert(sizeof(bool) == sizeof(char));
    return *reinterpret_cast<const unsigned char*>(src);
  }
};

template <typename T>
HOSTDEVICE constexpr T LoadData(const void* src) {
  return LoadImpl<T>::Apply(src);
}

template <typename ScalarT>
HOSTDEVICE constexpr ScalarT LoadData(const ScalarT* src) {
  return LoadImpl<ScalarT>::Apply(src);
}

namespace phi {
inline std::bitset<64> DimListToBitset(std::vector<int> opt_dims,
                                       size_t ndims) {
  std::bitset<64> dim_mask;

  if (opt_dims.size() > 0) {
    for (int dim : opt_dims) {
      dim_mask.set(dim, true);
    }
  } else {
    for (size_t dim = 0; dim < ndims; dim++) {
      dim_mask.set(dim, true);
    }
  }
  return dim_mask;
}

inline std::vector<int> ConvertToPositiveDims(
    const std::vector<int>& origin_reduce_dims, int64_t ndim) {
  std::vector<int> positive_reduce_dims = origin_reduce_dims;
  for (size_t i = 0; i < origin_reduce_dims.size(); ++i) {
    PADDLE_ENFORCE_GE(
        origin_reduce_dims[i],
        -ndim,
        common::errors::InvalidArgument(
            "ReduceOp: invalid axis, when x_dims is %d, "
            "axis[i] should be in the range of [-%d, %d), but got %d.",
            ndim,
            ndim,
            ndim,
            origin_reduce_dims[i]));
    PADDLE_ENFORCE_LT(
        origin_reduce_dims[i],
        ndim,
        common::errors::InvalidArgument(
            "ReduceOp: invalid axis, when x_dims is %d, "
            "axis[i] should be in the range of [-%d, %d), but got %d.",
            ndim,
            ndim,
            ndim,
            origin_reduce_dims[i]));

    if (origin_reduce_dims[i] < 0) {
      positive_reduce_dims[i] = ndim + origin_reduce_dims[i];
    }
  }
  return positive_reduce_dims;
}

inline std::bitset<64> MakeDimMask(std::vector<int> opt_dims,
                                   int64_t ndim,
                                   bool allow_empty_dims = false) {
  // flip() sets all bits to 1 (masking all dimensions for reduction).
  if (opt_dims.empty() && !allow_empty_dims) {
    return std::bitset<64>().flip();
  }

  // Otherwise, use the dimensions specified in opt_dims.
  return DimListToBitset(opt_dims, ndim);
}

inline DenseTensor ReviewReduceResult(const DenseTensor& src,
                                      const DenseTensor& result,
                                      int ndim,
                                      std::bitset<64> mask) {
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;

  int64_t cal_stride = 1;
  const auto& src_dims = src.dims();

  for (int dim = ndim - 1; dim >= 0; dim--) {
    if (!mask[dim]) {
      shape.insert(shape.begin(), src_dims[dim]);
      stride.insert(stride.begin(), cal_stride);
      cal_stride *= src_dims[dim];
    } else {
      shape.insert(shape.begin(), 1);
      stride.insert(stride.begin(), cal_stride);
    }
  }

  return funcs::as_strided(result, shape, stride);
}

template <typename T, int Size>
DEVICE AlignedVector<T, Size> LoadVector(const T* base_ptr, uint32_t offset) {
  using vec_t = AlignedVector<T, Size>;
  auto* from = reinterpret_cast<const vec_t*>(base_ptr);
  return from[offset];
}

template <int Size>
DEVICE AlignedVector<bool, Size> LoadVector(const bool* base_ptr,
                                            uint32_t offset) {
  auto tmp = LoadVector<uint8_t, Size>(
      reinterpret_cast<const uint8_t*>(base_ptr), offset);
  AlignedVector<bool, Size> ret;
  for (int i = 0; i < Size; ++i) {
    ret.val[i] = static_cast<bool>(tmp.val[i]);
  }
  return ret;
}

// Chose max num threads.
template <typename T>
struct MaxThreadsConfig {
  static constexpr int MAX_NUM_THREADS = 512;
};

template <>
struct MaxThreadsConfig<phi::dtype::complex<double>> {
  static constexpr int MAX_NUM_THREADS = 256;
};

template <int kNumThreads, int kOutputVecSize, typename Reducer>
__launch_bounds__(kNumThreads, 4) __global__
    void VecReduceKernel(Reducer reduction) {
  reduction.template Run<kOutputVecSize>();
}

template <typename IndexType>
static funcs::OffsetCalculator<2, IndexType> MakeOutputOffsetCalculator(
    const DenseTensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int num_output_dims = iter.ndim() - num_reduce_dims;
  int input_index = iter.ntensors() - 1;
  int output_index = 0;

  std::array<const int64_t*, 2> stride_ptrs = {
      iter.strides(output_index).data() + num_reduce_dims,
      iter.strides(input_index).data() + num_reduce_dims,
  };

  auto output_shape_ptr = iter.shape().data() + num_reduce_dims;

  return funcs::OffsetCalculator<2, IndexType>(
      num_output_dims, output_shape_ptr, stride_ptrs.data());
}

template <typename IndexType>
static funcs::OffsetCalculator<1, IndexType> MakeInputOffsetCalculator(
    const DenseTensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int input_index = iter.ntensors() - 1;

  std::array<const int64_t*, 1> strides = {
      iter.strides(input_index).data(),
  };

  auto input_shape_ptr = iter.shape().data();

  return funcs::OffsetCalculator<1, IndexType>(
      num_reduce_dims, input_shape_ptr, strides.data());
}

template <typename T>
int GetOutputVecSize(const DenseTensorIterator& iter) {
  int vec_size = 4;

  auto UpdateVectorSize = [&vec_size](uint64_t n) {
    while (n % vec_size != 0) {
      vec_size /= 2;
    }
  };

  // Check base address alignment.
  uint64_t base_address =
      reinterpret_cast<uint64_t>(iter.data_ptr(iter.noutputs())) / sizeof(T);
  UpdateVectorSize(base_address);

  // Check output dimension size.
  const int output_index = iter.num_reduce_dims();
  UpdateVectorSize(iter.shape()[output_index]);

  // Check strides alignment for all dimensions except output dimension.
  auto input_tensor_index = iter.noutputs();
  auto input_strides = iter.strides(input_tensor_index);

  for (int dim = 0; dim < input_strides.size(); ++dim) {
    if (dim != output_index) {
      UpdateVectorSize(input_strides[dim] / sizeof(T));
    }
  }

  return vec_size;
}

// Simplify fraction by dividing both numerator and denominator by their GCD
// (Greatest Common Divisor).
HOSTDEVICE static void ReduceFraction(size_t* numerator, size_t* denominator) {
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

  // Basic configuration.
  int element_size_bytes;
  int num_inputs;
  int num_outputs;

  // Parallelism control.
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;

  // Multiplier arrays for index calculation.
  int input_multiplier[3] = {0, 0, 0};
  int output_multiplier[2] = {0, 0};

  // Dimensions.
  int block_width;
  int block_height;
  int num_threads;

  // Vectorization control.
  bool vectorize_input = false;
  int output_vec_size = 1;

  template <typename T>
  void SetBlockDimensions(int64_t dim0, int64_t dim1) {
    const int max_num_threads =
        MaxThreadsConfig<T>::MAX_NUM_THREADS / output_vec_size;

    int dim0_pow2 =
        (dim0 < max_num_threads)
            ? static_cast<int>(phi::backends::gpu::GetLastPow2(dim0))
            : max_num_threads;
    int dim1_pow2 =
        (dim1 < max_num_threads)
            ? static_cast<int>(phi::backends::gpu::GetLastPow2(dim1))
            : max_num_threads;
    block_width = std::min(dim0_pow2, WARP_SIZE);
    block_height =
        std::min(dim1_pow2, static_cast<int>(max_num_threads / block_width));
    block_width =
        std::min(dim0_pow2, static_cast<int>(max_num_threads / block_height));
    num_threads = block_width * block_height;
  }

  int SplitInput(int parallelism) {
    const int current_step = step_input;
    step_input *= parallelism;
    return current_step;
  }

  int SplitOutput(int parallelism) {
    const int current_step = step_output;
    step_output *= parallelism;
    return current_step;
  }

  dim3 GetBlockDim() const { return dim3(block_width, block_height); }

  dim3 GetGridDim() const {
    return dim3(phi::backends::gpu::DivUp<int64_t>(
                    num_outputs / output_vec_size, step_output),
                ctas_per_output);
  }

  HOSTDEVICE bool ShouldReduceBlockX() const {
    return input_multiplier[BLOCK_X] != 0;
  }

  HOSTDEVICE bool ShouldReduceBlockY() const {
    return input_multiplier[BLOCK_Y] != 0;
  }

  HOSTDEVICE bool ShouldReduceGlobal() const {
    return input_multiplier[CTA] != 0;
  }

  DEVICE bool ShouldStore(int output_idx) const {
    // 1. Boundary Check: Ensure the output index is within the valid range.
    //    If out of bounds, no storage is necessary.
    if (output_idx >= num_outputs) {
      return false;
    }

    // 2. X-Reduction Check: If block-wide X-reduction is active, only the
    //    thread with index 0 in the X-dimension (the "leader") is allowed to
    //    store.
    if (ShouldReduceBlockX() && threadIdx.x != 0) {
      return false;
    }

    // 3. Y-Reduction Check: If block-wide Y-reduction is active, only the
    //    thread with index 0 in the Y-dimension (the "leader") is allowed to
    //    store.
    if (ShouldReduceBlockY() && threadIdx.y != 0) {
      return false;
    }

    // If the thread passes all checks, it is the designated thread to store the
    // result.
    return true;
  }

  DEVICE bool ShouldReduceTail() const {
    return (!ShouldReduceBlockY() || threadIdx.y == 0) &&
           (!ShouldReduceGlobal() || blockIdx.y == 0);
  }

  HOSTDEVICE int GetInIdx() const {
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int block_y = blockIdx.y;
    return (thread_x * input_multiplier[BLOCK_X] +
            thread_y * input_multiplier[BLOCK_Y] +
            block_y * input_multiplier[CTA]);
  }

  template <int kOutputVecSize>
  HOSTDEVICE int GetOutIdx() const {
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int block_x = blockIdx.x;
    return (thread_x * output_multiplier[BLOCK_X] +
            thread_y * output_multiplier[BLOCK_Y] + block_x * step_output) *
           kOutputVecSize;
  }

  DEVICE int SharedMemoryOffset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  DEVICE int StagingMemoryOffset(int block_y) const {
    IndexType offset = block_y + static_cast<IndexType>(blockIdx.x) *
                                     static_cast<IndexType>(gridDim.y);
    if (!ShouldReduceBlockX()) {
      offset = threadIdx.x + offset * blockDim.x;
    }

    return offset;
  }

  int SharedMemorySize() const {
    if (!ShouldReduceBlockY() &&
        (!ShouldReduceBlockX() || block_width <= WARP_SIZE)) {
      return 0;
    }

    return element_size_bytes * num_threads * output_vec_size;
  }

  int64_t GlobalMemorySize() const {
    if (!ShouldReduceGlobal()) {
      return 0;
    }

    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!ShouldReduceBlockX()) {
      size *= GetBlockDim().x * output_vec_size;
    }

    return size;
  }

  int SemaphoreSize() const {
    if (!ShouldReduceGlobal()) {
      return 0;
    }
    return sizeof(int) * GetGridDim().x;
  }

  int ValuesPerThread() const {
    return phi::backends::gpu::DivUp<int64_t>(num_inputs, step_input);
  }
};

template <typename MPType,
          typename ScalarT,
          int kVecSize,
          int kInputVecSize = kVecSize>
ReduceConfig SetReduceConfig(const DenseTensorIterator& iter) {
  int device_id = paddle::platform::GetCurrentDeviceId();

  int64_t num_outputs = iter.num_output_elements();
  int64_t inputs_per_output = iter.numel() / num_outputs;
  int input_index = iter.ntensors() - 1;

  auto config = ReduceConfig(sizeof(MPType), num_outputs, inputs_per_output);

  int64_t dim0;
  int64_t dim1;
  int64_t fastest_moving_stride;
  bool reduce_fastest_dim;

  if (iter.ndim() > 0) {
    // Check if we're reducing along the fastest-changing dimension
    // This affects memory access patterns for better performance.
    reduce_fastest_dim = (iter.num_reduce_dims() == iter.ndim()) ||
                         (iter.strides(input_index)[0] <
                          iter.strides(input_index)[iter.num_reduce_dims()]);

    // Set block dimensions based on reduction pattern.
    if (reduce_fastest_dim) {
      // Reducing along fastest dimension: use block.x for reduction.
      //    block.x handles reduction elements.
      //    block.y handles output elements.
      dim0 = inputs_per_output;
      dim1 = num_outputs;
      fastest_moving_stride = iter.strides(input_index)[0];
    } else {
      // Not reducing along fastest dimension: use block.x for outputs.
      //    block.x handles output elements.
      //    block.y handles reduction elements.
      dim0 = num_outputs;
      dim1 = inputs_per_output;
      fastest_moving_stride = iter.strides(input_index)[iter.num_reduce_dims()];
    }
  } else {
    // Handle 0-dimensional case.
    reduce_fastest_dim = true;
    fastest_moving_stride = sizeof(ScalarT);
    dim0 = 1;
    dim1 = 1;
  }

  // Use vectorization for better memory access. Two cases:
  // Case 1: "Vectorize along input" - when reducing on fastest dimension,
  //         data in same vector corresponds to the same output.
  // Case 2: "Vectorize along output" - when fastest dimension is not reduced,
  //         data in same vector corresponds to different outputs.
  if (fastest_moving_stride == sizeof(ScalarT)) {
    if (reduce_fastest_dim && dim0 > 128 && iter.num_reduce_dims() == 1 &&
        kVecSize >= kInputVecSize) {
      // Case 1: Vectorize along input (load data for same output together).
      config.vectorize_input = true;
      dim0 /= kInputVecSize;
    } else if (!reduce_fastest_dim) {
      // Case 2: Vectorize along output (load data for multiple outputs
      // together).
      config.output_vec_size = GetOutputVecSize<ScalarT>(iter);
      dim0 /= config.output_vec_size;
    }
  }

  // Adjust block_width and block_height.
  config.SetBlockDimensions<ScalarT>(dim0, dim1);

  int block_width = config.block_width;
  int block_height = config.block_height;

  // Level 1 parallelization: split work at thread level.
  if (iter.ndim() == 0 || reduce_fastest_dim) {
    // Case 1: Split input across threads (requires thread synchronization).
    config.input_multiplier[0] = config.SplitInput(block_width);
  } else {
    // Case 2: Split output across threads (each thread handles different
    // output).
    config.output_multiplier[0] = config.SplitOutput(block_width);
  }

  // Min elements per thread.
  constexpr int min_values_per_thread = 16;
  // Max elements per thread.
  constexpr int max_values_per_thread = 256;

  // Decide if we need to split work across warps.
  const int warp_split_threshold =
      std::min<int>(block_height * 16, max_values_per_thread);
  bool split_across_warps = config.ValuesPerThread() >= warp_split_threshold;

  const int num_mp = paddle::platform::GetGPUMultiProcessors(device_id);

  // Level 2 parallelization: split work at warp level.
  if (split_across_warps) {
    // Case 1: Split input across warps (requires warp synchronization).
    config.input_multiplier[1] = config.SplitInput(block_height);
  } else {
    // Case 2: Each warp handles independent outputs.
    config.output_multiplier[1] = config.SplitOutput(block_height);
  }

  int max_threads_per_mp =
      paddle::platform::GetGPUMaxThreadsPerMultiProcessor(device_id);

  const int blocks_per_sm = max_threads_per_mp / config.num_threads;
  const int target_grid_size = num_mp * blocks_per_sm;
  int grid = config.GetGridDim().x;

  // Level 3 parallelization: split work at block level (for large datasets).
  if (config.input_multiplier[1] != 0 &&
      config.ValuesPerThread() >= max_values_per_thread &&
      grid <= target_grid_size) {
    // Calculate optimal block splitting strategy.
    // Based on SM utilization.
    int ctas_per_output1 =
        phi::backends::gpu::DivUp<int64_t>(target_grid_size, grid);
    // Based on min workload.
    int ctas_per_output2 = phi::backends::gpu::DivUp<int64_t>(
        config.ValuesPerThread(), min_values_per_thread);
    // Based on max workload.
    int ctas_per_output3 = phi::backends::gpu::DivUp<int64_t>(
        config.ValuesPerThread(), max_values_per_thread);

    // Choose best splitting strategy to balance parallelism and per-thread
    // workload.
    config.ctas_per_output = std::max(
        std::min<int>(ctas_per_output1, ctas_per_output2), ctas_per_output3);

    if (config.ctas_per_output > 1) {
      // Case 3: Split input across blocks (requires global memory
      // synchronization).
      config.input_multiplier[2] = config.SplitInput(config.ctas_per_output);
    }
  }
  return config;
}

template <typename ScalarT,
          typename ReduceOp,
          typename OutScalarT = ScalarT,
          int kVecSize = 4,
          int kInputVecSize = kVecSize>
struct ReduceExecutor {
  using traits = phi::funcs::FunctionTraits<decltype(&ReduceOp::reduce)>;
  using MPType =
      typename std::decay<typename traits::template arg<0>::type>::type;

  using InputCalculator = funcs::OffsetCalculator<1, IndexType>;
  using OutputCalculator = funcs::OffsetCalculator<2, IndexType>;

  static constexpr bool can_accumulate_in_output =
      std::is_convertible_v<MPType, OutScalarT> &&
      std::is_convertible_v<OutScalarT, MPType>;

  // Core reduction algorithm configuration.
  ReduceOp reducer;
  ReduceConfig config;
  MPType ident;

  // Data access calculators for input and output indexing.
  InputCalculator input_calc;
  OutputCalculator output_calc;

  // Data pointers for source, destination, and buffers.
  const void* src;
  char* dst[2];
  void* acc_buf;
  void* cta_buf;

  // Parallel synchronization primitives.
  int* semaphores;

  // Runtime state and control flags.
  int64_t base_idx;
  bool accumulate;
  bool final_output;
  int noutputs;

  ReduceExecutor(ReduceOp reducer,
                 ReduceConfig config,
                 MPType ident,
                 InputCalculator input_calc,
                 OutputCalculator output_calc,
                 const void* src,
                 char* dst0,
                 std::optional<char*> dst1,
                 void* acc_buf,
                 void* cta_buf,
                 int* semaphores,
                 int base_idx,
                 bool accumulate,
                 bool final_output,
                 int64_t noutputs)
      : reducer(reducer),
        config(config),
        ident(ident),
        input_calc(input_calc),
        output_calc(output_calc),
        src(src),
        acc_buf(acc_buf),
        cta_buf(cta_buf),
        semaphores(semaphores),
        base_idx(base_idx),
        accumulate(accumulate),
        final_output(final_output),
        noutputs(noutputs) {
    dst[0] = dst0;
    if (dst1.has_value()) {
      dst[1] = dst1.value();
    }
  }

  template <int kOutputVecSize>
  DEVICE void Run() const {
    extern __shared__ char shared_memory[];

    IndexType output_idx = config.GetOutIdx<kOutputVecSize>();
    IndexType input_idx = config.GetInIdx();
    auto base_offsets1 = output_calc.get(output_idx)[1];

    using MPTypeVec = std::array<MPType, kOutputVecSize>;
    MPTypeVec value;

    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      const ScalarT* input_slice =
          (const ScalarT*)((const char*)src + base_offsets1);
      value = ThreadReduce<kOutputVecSize>(input_slice);
    }

    if (config.ShouldReduceBlockY()) {
      value = BlockYReduce<kOutputVecSize>(value, shared_memory);
    }

    if (config.ShouldReduceBlockX()) {
      value = BlockXReduce<kOutputVecSize>(value, shared_memory);
    }

    using OutPtrVec = std::array<OutScalarT*, kOutputVecSize>;
    using OffsetVec = std::array<IndexType, kOutputVecSize>;

    OffsetVec base_offsets;
    OutPtrVec out;

#pragma unroll
    for (int i = 0; i < kOutputVecSize; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = reinterpret_cast<OutScalarT*>(dst[0] + base_offsets[i]);
    }

    MPTypeVec* acc = nullptr;
    if (acc_buf != nullptr) {
      size_t numerator = sizeof(MPType);
      size_t denominator = sizeof(OutScalarT);
      ReduceFraction(&numerator, &denominator);
      acc = reinterpret_cast<MPTypeVec*>(
          reinterpret_cast<char*>(acc_buf) +
          (base_offsets[0] * numerator / denominator));
    }

    if (config.ShouldReduceGlobal()) {
      value = GlobalReduce<kOutputVecSize>(value, acc, shared_memory);
    } else if (config.ShouldStore(output_idx)) {
      if (acc == nullptr) {
        if (accumulate) {
          value = AccumulateInOutput<kOutputVecSize, can_accumulate_in_output>(
              out, value);
        }
        if (final_output) {
          SetResultsToOutput<kOutputVecSize>(value, base_offsets);
        } else {
#pragma unroll
          for (int i = 0; i < kOutputVecSize; i++) {
            *(out[i]) = GetAccumulatedOutput<can_accumulate_in_output>(
                out[i], value[i]);
          }
        }
      } else {
        if (accumulate) {
#pragma unroll
          for (int i = 0; i < kOutputVecSize; i++) {
            value[i] = reducer.reduce((*acc)[i], value[i]);
          }
        }
        if (final_output) {
          SetResultsToOutput<kOutputVecSize>(value, base_offsets);
        } else {
          *acc = value;
        }
      }
    }
  }

  template <int kOutputVecSize>
  DEVICE std::array<MPType, kOutputVecSize> ThreadReduce(
      const ScalarT* data) const {
    if (config.vectorize_input) {
      return {InVectorizedThreadReduceImpl(data)};
    } else {
      IndexType element_stride = input_calc.strides_[0][0] / sizeof(ScalarT);
      bool is_contiguous = (input_calc.dims == 1 && element_stride == 1);
      if (is_contiguous) {
        return ThreadReduceImpl<kOutputVecSize>(
            data, [](IndexType idx) { return idx; });
      } else if (input_calc.dims == 1) {
        return ThreadReduceImpl<kOutputVecSize>(
            data, [&](IndexType idx) { return idx * element_stride; });
      } else {
        return ThreadReduceImpl<kOutputVecSize>(data, [&](IndexType idx) {
          return input_calc.get(idx)[0] / sizeof(ScalarT);
        });
      }
    }
  }

  DEVICE MPType InVectorizedThreadReduceImpl(const ScalarT* data) const {
    IndexType end = config.num_inputs;
    MPType value = ident;
    constexpr int align_bytes =
        alignof(phi::AlignedVector<ScalarT, kInputVecSize>);

    constexpr int align_elements = align_bytes / sizeof(ScalarT);
    int shift = ((uint64_t)data) % align_bytes / sizeof(ScalarT);

    if (shift > 0) {
      data -= shift;
      end += shift;
      if (threadIdx.x >= shift && threadIdx.x < align_elements &&
          config.ShouldReduceTail()) {
        value = reducer.compute(value, LoadData(data + threadIdx.x));
      }
      end -= align_elements;
      data += align_elements;
      shift = align_elements - shift;
    }

    IndexType idx = config.GetInIdx();
    const IndexType stride = config.step_input;

    std::array<MPType, kInputVecSize> value_list;
    value_list[0] = value;

#pragma unroll
    for (int i = 1; i < kInputVecSize; i++) {
      value_list[i] = ident;
    }

    using load_t = phi::AlignedVector<ScalarT, kInputVecSize>;

    while (idx * kInputVecSize + kInputVecSize - 1 < end) {
      const auto values_vec = LoadVector<ScalarT, kInputVecSize>(data, idx);

#pragma unroll
      for (IndexType i = 0; i < kInputVecSize; i++) {
        value_list[i] = reducer.compute(value_list[i], values_vec.val[i]);
      }
      idx += stride;
    }

    // Tile processing.
    IndexType tail_start = end - end % kInputVecSize;

    if (config.ShouldReduceTail()) {
      int idx = tail_start + threadIdx.x;
      if (idx < end) {
        const auto value = LoadData(data + idx);
        value_list[0] = reducer.compute(value_list[0], value);
      }
    }

#pragma unroll
    for (int i = 1; i < kInputVecSize; i++) {
      value_list[0] = reducer.reduce(value_list[0], value_list[i]);
    }

    return value_list[0];
  }

  template <int kOutputVecSize, typename offset_calc_t>
  DEVICE std::array<MPType, kOutputVecSize> ThreadReduceImpl(
      const ScalarT* data_, offset_calc_t calc) const {
    IndexType idx = config.GetInIdx();
    const IndexType end = config.num_inputs;
    const IndexType stride = config.step_input;

    using MPTypeVec = std::array<MPType, kOutputVecSize>;
    using load_t = phi::AlignedVector<ScalarT, kOutputVecSize>;

    std::array<MPTypeVec, kVecSize> value_list;

#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
#pragma unroll
      for (int j = 0; j < kOutputVecSize; j++) {
        value_list[i][j] = ident;
      }
    }

    std::array<load_t, kVecSize> values;

    while (idx + (kVecSize - 1) * stride < end) {
#pragma unroll
      for (IndexType i = 0; i < kVecSize; i++) {
        const auto offset = calc(idx + i * stride) / kOutputVecSize;
        values[i] = LoadVector<ScalarT, kOutputVecSize>(data_, offset);
      }
#pragma unroll
      for (IndexType i = 0; i < kVecSize; i++) {
#pragma unroll
        for (IndexType j = 0; j < kOutputVecSize; j++) {
          value_list[i][j] =
              reducer.compute(value_list[i][j], values[i].val[j]);
        }
      }
      idx += stride * kVecSize;
    }

    // tail
    int idx_ = idx;
#pragma unroll
    for (IndexType i = 0; i < kVecSize; i++) {
      if (idx >= end) {
        break;
      }
      const auto offset = calc(idx) / kOutputVecSize;
      values[i] = LoadVector<ScalarT, kOutputVecSize>(data_, offset);
      idx += stride;
    }
    idx = idx_;
#pragma unroll
    for (IndexType i = 0; i < kVecSize; i++) {
      if (idx >= end) {
        break;
      }
#pragma unroll
      for (IndexType j = 0; j < kOutputVecSize; j++) {
        value_list[i][j] = reducer.compute(value_list[i][j], values[i].val[j]);
      }
      idx += stride;
    }

#pragma unroll
    for (int i = 1; i < kVecSize; i++) {
#pragma unroll
      for (IndexType j = 0; j < kOutputVecSize; j++) {
        value_list[0][j] = reducer.reduce(value_list[0][j], value_list[i][j]);
      }
    }
    return value_list[0];
  }

  template <int kOutputVecSize>
  DEVICE std::array<MPType, kOutputVecSize> BlockXReduce(
      std::array<MPType, kOutputVecSize> value, char* shared_memory) const {
    using MPTypeVec = std::array<MPType, kOutputVecSize>;
    int dim_x = blockDim.x;
    MPTypeVec* shared = reinterpret_cast<MPTypeVec*>(shared_memory);

    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);

    if (dim_x > WARP_SIZE) {
      IndexType address_base = static_cast<IndexType>(threadIdx.x) +
                               static_cast<IndexType>(threadIdx.y) *
                                   static_cast<IndexType>(blockDim.x);

      shared[address_base] = value;
      for (int offset = dim_x / 2; offset >= WARP_SIZE; offset >>= 1) {
        __syncthreads();

        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
          MPTypeVec other = shared[address_base + offset];
#pragma unroll
          for (int i = 0; i < kOutputVecSize; i++) {
            value[i] = reducer.reduce(value[i], other[i]);
          }
          shared[address_base] = value;
        }
      }
      dim_x = WARP_SIZE;
    }

    __syncthreads();

    for (int offset = 1; offset < dim_x; offset <<= 1) {
#pragma unroll
      for (int i = 0; i < kOutputVecSize; i++) {
        MPType other = reducer.shfl_sync(mask, value[i], offset);
        value[i] = reducer.reduce(value[i], other);
      }
    }
    return value;
  }

  template <int kOutputVecSize>
  DEVICE std::array<MPType, kOutputVecSize> BlockYReduce(
      std::array<MPType, kOutputVecSize> value, char* shared_memory) const {
    using MPTypeVec = std::array<MPType, kOutputVecSize>;
    MPTypeVec* shared = reinterpret_cast<MPTypeVec*>(shared_memory);
    shared[config.SharedMemoryOffset(0)] = value;

    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
        MPTypeVec other = shared[config.SharedMemoryOffset(offset)];
#pragma unroll
        for (int i = 0; i < kOutputVecSize; i++) {
          value[i] = reducer.reduce(value[i], other[i]);
        }
        shared[config.SharedMemoryOffset(0)] = value;
      }
    }
    return value;
  }

  DEVICE bool MarkBlockFinished() const {
    __shared__ bool is_last_block_done_shared;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
    }

    __syncthreads();

    return is_last_block_done_shared;
  }

  template <int kOutputVecSize, bool can_acc>
  DEVICE std::array<MPType, kOutputVecSize> AccumulateInOutput(
      std::array<OutScalarT*, kOutputVecSize> out,
      std::array<MPType, kOutputVecSize> value) const {
    if constexpr (can_acc) {
      std::array<MPType, kOutputVecSize> ret;
#pragma unroll
      for (int i = 0; i < kOutputVecSize; i++) {
        ret[i] = reducer.reduce(*(out[i]), value[i]);
      }
      return ret;
    } else {
      return {MPType{}};
    }
  }

  template <bool can_acc>
  DEVICE OutScalarT GetAccumulatedOutput(OutScalarT* out, MPType value) const {
    if constexpr (can_acc) {
      return (OutScalarT)value;
    } else {
      return *out;
    }
  }

  template <class T>
  DEVICE void SetResults(const T x, const IndexType base_offset) const {
    auto res = reinterpret_cast<OutScalarT*>(dst[0] + base_offset);
    *res = x;
  }

  template <class T1, class T2>
  DEVICE void SetResults(const thrust::pair<T1, T2> x,
                         const IndexType base_offset) const {
    if (noutputs >= 1) {
      auto res0 = reinterpret_cast<T1*>(dst[0] + base_offset);
      *res0 = x.first;
    }
    if (noutputs >= 2) {
      auto res1 =
          reinterpret_cast<T2*>(dst[1] + base_offset / sizeof(T1) * sizeof(T2));
      *res1 = x.second;
    }
  }

  template <int kOutputVecSize>
  DEVICE void SetResultsToOutput(
      std::array<MPType, kOutputVecSize> value,
      std::array<IndexType, kOutputVecSize> base_offset) const {
#pragma unroll
    for (int i = 0; i < kOutputVecSize; i++) {
      SetResults(reducer.post_process(value[i]), base_offset[i]);
    }
  }

  template <int kOutputVecSize>
  DEVICE std::array<MPType, kOutputVecSize> GlobalReduce(
      std::array<MPType, kOutputVecSize> value,
      std::array<MPType, kOutputVecSize>* acc,
      char* shared_memory) const {
    using MPTypeVec = std::array<MPType, kOutputVecSize>;
    using OutPtrVec = std::array<OutScalarT*, kOutputVecSize>;
    using OffsetVec = std::array<IndexType, kOutputVecSize>;

    MPTypeVec* reduce_buffer = reinterpret_cast<MPTypeVec*>(cta_buf);
    IndexType output_idx = config.GetOutIdx<kOutputVecSize>();
    OffsetVec base_offsets;
    OutPtrVec out;

#pragma unroll
    for (int i = 0; i < kOutputVecSize; i++) {
      base_offsets[i] = output_calc.get(output_idx + i)[0];
      out[i] = reinterpret_cast<OutScalarT*>(dst[0] + base_offsets[i]);
    }

    bool should_store = config.ShouldStore(output_idx);
    if (should_store) {
      IndexType offset = config.StagingMemoryOffset(blockIdx.y);
      reduce_buffer[offset] = value;
    }

    __threadfence();

    __syncthreads();

    bool is_last_block_done = MarkBlockFinished();

    if (is_last_block_done) {
      __threadfence();

      for (auto& v : value) {
        v = ident;
      }

      if (config.ShouldReduceBlockX()) {
        IndexType input_offset = static_cast<IndexType>(threadIdx.x) +
                                 static_cast<IndexType>(threadIdx.y) *
                                     static_cast<IndexType>(blockDim.x);
        IndexType step = static_cast<IndexType>(blockDim.x) *
                         static_cast<IndexType>(blockDim.y);

        for (; input_offset < config.ctas_per_output; input_offset += step) {
          IndexType idx = config.StagingMemoryOffset(input_offset);
          MPTypeVec next = reduce_buffer[idx];
#pragma unroll
          for (int i = 0; i < kOutputVecSize; i++) {
            value[i] = reducer.reduce(value[i], next[i]);
          }
        }
      } else {
        IndexType input_offset = threadIdx.y;
        IndexType step = blockDim.y;

        for (; input_offset < config.ctas_per_output; input_offset += step) {
          IndexType idx = config.StagingMemoryOffset(input_offset);
          MPTypeVec next = reduce_buffer[idx];
#pragma unroll
          for (int i = 0; i < kOutputVecSize; i++) {
            value[i] = reducer.reduce(value[i], next[i]);
          }
        }
      }
      value = BlockYReduce<kOutputVecSize>(value, shared_memory);
      if (config.ShouldReduceBlockX()) {
        value = BlockXReduce<kOutputVecSize>(value, shared_memory);
      }
      if (should_store) {
        if (acc == nullptr) {
          if (accumulate) {
            value =
                AccumulateInOutput<kOutputVecSize, can_accumulate_in_output>(
                    out, value);
          }
          if (final_output) {
            SetResultsToOutput<kOutputVecSize>(value, base_offsets);
          } else {
#pragma unroll
            for (int i = 0; i < kOutputVecSize; i++) {
              *(out[i]) = GetAccumulatedOutput<can_accumulate_in_output>(
                  out[i], value[i]);
            }
          }
        } else {
          if (accumulate) {
#pragma unroll
            for (int i = 0; i < kOutputVecSize; i++) {
              value[i] = reducer.reduce((*acc)[i], value[i]);
            }
          }
          if (final_output) {
            SetResultsToOutput<kOutputVecSize>(value, base_offsets);
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
      acc_ptr_ = reinterpret_cast<char*>(out_ptr);
      numerator_ = 1;
      denominator_ = 1;
    } else {
      phi::Allocator* allocator =
          const_cast<phi::Allocator*>(&(dev_ctx.GetAllocator()));  // NOLINT
      buffer_ = allocator->Allocate(size);
      acc_ptr_ = reinterpret_cast<char*>(buffer_->ptr());
      numerator_ = acc_t_size;
      denominator_ = out_t_size;
      ReduceFraction(&numerator_, &denominator_);
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
static void LaunchReduceKernel(const KPDevice& dev_ctx,
                               const ReduceConfig& config,
                               const R& reduction) {
  dim3 block = config.GetBlockDim();
  dim3 grid = config.GetGridDim();
  int shared_memory = config.SharedMemorySize();

  auto stream = dev_ctx.stream();

  switch (config.output_vec_size) {
    case 4:
      VecReduceKernel<max_threads / 4, 4, R>
          <<<grid, block, shared_memory, stream>>>(reduction);
      break;
    case 2:
      VecReduceKernel<max_threads / 2, 2, R>
          <<<grid, block, shared_memory, stream>>>(reduction);
      break;
    default:
      VecReduceKernel<max_threads / 1, 1, R>
          <<<grid, block, shared_memory, stream>>>(reduction);
      break;
  }
}

template <typename Tx,
          typename Ty,
          int kVecSize = 4,
          int kInputVecSize = kVecSize,
          typename ReduceOp,
          typename ident_t = double>
inline void GPUReduceScheduler(const KPDevice& dev_ctx,
                               const DenseTensorIterator& iter,
                               const ReduceOp& reducer,
                               ident_t ident = 0,
                               AccumulationBuffer* acc_buf_ptr = nullptr,
                               int64_t base_idx = 0) {
  auto stream = dev_ctx.stream();

  using traits = phi::funcs::FunctionTraits<decltype(&ReduceOp::reduce)>;
  using MPType = typename traits::template arg<0>::type;

  static constexpr bool is_inp_out_type_half_or_chalf =
      (std::is_same_v<phi::float16, Tx> && std::is_same_v<phi::float16, Ty>) ||
      (std::is_same_v<phi::dtype::complex<float16>, Tx> &&
       std::is_same_v<phi::dtype::complex<float16>, Ty>);
  static constexpr bool is_inp_out_type_bfloat16 =
      (std::is_same_v<phi::bfloat16, Tx> && std::is_same_v<phi::bfloat16, Ty>);
  static constexpr bool can_accumulate_in_output =
      std::is_convertible_v<MPType, Ty> &&
      !(is_inp_out_type_half_or_chalf || is_inp_out_type_bfloat16);

  bool can_use_32bit_indexing = iter.can_use_32bit_indexing();
  std::unique_ptr<AccumulationBuffer> owned_buf_ptr;
  if (acc_buf_ptr == NULL) {
    if (!can_accumulate_in_output && !can_use_32bit_indexing) {
      int64_t output_memory_size = phi::SizeOf(iter.dtype(0));
      for (int dim = 0; dim < iter.ndim(); dim++) {
        output_memory_size = std::max(output_memory_size,
                                      iter.shape()[dim] * iter.strides(0)[dim]);
      }
      output_memory_size /= phi::SizeOf(iter.dtype(0));
      owned_buf_ptr.reset(
          new AccumulationBuffer(dev_ctx,
                                 sizeof(MPType),
                                 sizeof(Ty),
                                 reinterpret_cast<char*>(iter.data_ptr(0)),
                                 output_memory_size * sizeof(MPType)));
    } else {
      owned_buf_ptr.reset(new AccumulationBuffer());
    }
    acc_buf_ptr = owned_buf_ptr.get();
  }

  // Split iter if index exceeds 32-bit range.
  if (!can_use_32bit_indexing) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      int64_t sub_iter_base_idx = sub_iter.view_offsets()[0];
      GPUReduceScheduler<Tx, Ty, kVecSize, kInputVecSize, ReduceOp>(
          dev_ctx, sub_iter, reducer, ident, acc_buf_ptr, sub_iter_base_idx);
    }
    return;
  }

  const char* in_data =
      reinterpret_cast<const char*>(iter.data_ptr(iter.ntensors() - 1));
  char* out_data = reinterpret_cast<char*>(iter.data_ptr(0));
  const auto noutputs = iter.noutputs();

  std::optional<char*> out_data_extra;
  if (noutputs > 1) {
    out_data_extra = reinterpret_cast<char*>(iter.data_ptr(1));
  } else {
    out_data_extra = std::nullopt;
  }

  char* acc_data = acc_buf_ptr->GetAccSlice(out_data);

  ReduceConfig config =
      SetReduceConfig<MPType, Tx, kVecSize, kInputVecSize>(iter);

  Allocator::AllocationPtr buffer;
  Allocator::AllocationPtr semaphores;
  void* buffer_ptr;
  void* semaphores_ptr;

  if (config.ShouldReduceGlobal()) {
    phi::Allocator* allocator =
        const_cast<phi::Allocator*>(&(dev_ctx.GetAllocator()));  // NOLINT
    buffer = allocator->Allocate(config.GlobalMemorySize());
    semaphores = allocator->Allocate(config.SemaphoreSize());
    buffer_ptr = buffer->ptr();
    semaphores_ptr = semaphores->ptr();

    phi::backends::gpu::GpuMemsetAsync(
        semaphores_ptr, 0, config.SemaphoreSize(), stream);
  }

  auto output_calc = MakeOutputOffsetCalculator<uint32_t>(iter);
  auto input_calc = MakeInputOffsetCalculator<uint32_t>(iter);
  auto should_accumulate = iter.should_accumulate();
  auto is_final_output = iter.is_final_output();

  auto reduce = ReduceExecutor<Tx, ReduceOp, Ty, kVecSize, kInputVecSize>(
      reducer,
      config,
      ident,
      input_calc,
      output_calc,
      in_data,
      out_data,
      out_data_extra,
      acc_data,
      buffer_ptr,
      reinterpret_cast<int*>(semaphores_ptr),
      base_idx,
      should_accumulate,
      is_final_output,
      noutputs);

  LaunchReduceKernel<MaxThreadsConfig<Tx>::MAX_NUM_THREADS>(
      dev_ctx, config, reduce);

  return;
}

namespace funcs {
template <typename Tx,
          typename Ty,
          template <typename, typename, typename>
          class ReduceOp>
void ReduceGpuKernel(const KPDevice& dev_ctx,
                     const phi::DenseTensor& x,
                     phi::DenseTensor* y,
                     const std::vector<int>& origin_reduce_dims,
                     const float norm_p = 1.0f) {
  if (x.numel() == 0) {
    dev_ctx.Alloc<Ty>(y);
    return;
  }

  dev_ctx.Alloc<Ty>(y);

  int64_t ndim = x.dims().size();
  auto positive_reduce_dims = ConvertToPositiveDims(origin_reduce_dims, ndim);
  auto mask = MakeDimMask(positive_reduce_dims, ndim);
  auto viewed_result = ReviewReduceResult(x, *(y), ndim, mask);

  auto x_dim = common::vectorize<int64_t>(x.dims());

  DenseTensorIteratorConfig dense_iter_config;
  dense_iter_config.is_reduction(true);
  dense_iter_config.add_output(viewed_result);
  dense_iter_config.add_const_input(x);
  DenseTensorIterator iter = dense_iter_config.build();

  // TODO(baoqiwen): When ReduceOp is WelfordOps, kVecSize is 2.
  constexpr int kVecSize = 4;
  constexpr int kInputVecSize = kVecSize;
  using MPType = typename phi::dtype::MPTypeTrait<Ty>::Type;

  // Initialize reducer.
  ReduceOp reducer = [&iter, &norm_p]() {
    constexpr bool kIsMeanOp =
        std::is_same_v<ReduceOp<Tx, MPType, Ty>, kps::MeanOps<Tx, MPType, Ty>>;

    constexpr bool kIsPNormOp =
        std::is_same_v<ReduceOp<Tx, MPType, Ty>,
                       kps::GenericPNormOps<Tx, MPType, Ty>>;

    if constexpr (kIsMeanOp) {
      MPType factor = static_cast<MPType>(iter.num_output_elements()) /
                      static_cast<MPType>(iter.numel());
      return ReduceOp<Tx, MPType, Ty>{factor};
    } else if constexpr (kIsPNormOp) {
      return ReduceOp<Tx, MPType, Ty>{norm_p};
    } else {
      return ReduceOp<Tx, MPType, Ty>{};
    }
  }();

  // Initialize ident value.
  Tx ident = []() {
    if constexpr (std::is_same_v<ReduceOp<Tx, MPType, Ty>,
                                 kps::MaxOps<Tx, MPType, Ty>> ||
                  std::is_same_v<ReduceOp<Tx, MPType, Ty>,
                                 kps::AbsMaxOps<Tx, MPType, Ty>>) {
      return std::numeric_limits<Tx>::lowest();
    }

    if constexpr (std::is_same_v<ReduceOp<Tx, MPType, Ty>,
                                 kps::MinOps<Tx, MPType, Ty>> ||
                  std::is_same_v<ReduceOp<Tx, MPType, Ty>,
                                 kps::AbsMinOps<Tx, MPType, Ty>>) {
      return std::numeric_limits<Tx>::max();
    }

    if constexpr (std::is_same_v<ReduceOp<Tx, MPType, Ty>,
                                 kps::LogicalAndOps<Tx, MPType, Ty>>) {
      return Tx{1};
    }

    if constexpr (std::is_same_v<ReduceOp<Tx, MPType, Ty>,
                                 kps::ProdOps<Tx, MPType, Ty>>) {
      return Tx{1};
    }

    // SumOps, MeanOps, LogicalOrOps and others
    return Tx{0};
  }();

  GPUReduceScheduler<Tx, Ty, kVecSize, kInputVecSize, ReduceOp<Tx, MPType, Ty>>(
      dev_ctx, iter, reducer, ident);

  return;
}
}  // namespace funcs
}  // namespace phi
