/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {
namespace funcs {

#ifdef PADDLE_WITH_HIP
constexpr int MAX_DIMS = 16;
#else
constexpr int MAX_DIMS = 25;
#endif

static constexpr int launch_bound2 = 4;

static constexpr int launch_size_nd = 128;

template <int nt, int vt, typename func_t>
__global__ void index_elementwise_kernel(const int64_t N, const func_t f) {
  const auto tid = threadIdx.x;
  const auto nv = nt * vt;
  auto idx = nv * blockIdx.x + tid;
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename Value>
struct DivMod {
  Value div, mod;

  __host__ __device__ DivMod(Value div, Value mod) : div(div), mod(mod) {}
};

template <typename Value>
struct IntDivider {
  IntDivider() = default;
  explicit IntDivider(Value d) : divisor(d) {}

  __host__ __device__ inline Value div(Value n) const { return n / divisor; }
  __host__ __device__ inline Value mod(Value n) const { return n % divisor; }
  __host__ __device__ inline DivMod<Value> divmod(Value n) const {
    return DivMod<Value>(n / divisor, n % divisor);
  }

  Value divisor;
};

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
  using stride_t =
      std::conditional_t<signed_strides, std::make_signed_t<index_t>, index_t>;
  using offset_type = std::array<stride_t, std::max<int>(NARGS, 1)>;

  OffsetCalculator(int dims,
                   const int64_t* sizes,
                   const int64_t* const* strides,
                   const int64_t* element_sizes = nullptr)
      : dims(dims) {
    PADDLE_ENFORCE(
        dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    for (int i = 0; i < dims; i++) {
      sizes_[i] = IntDivider<index_t>(sizes[i]);
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size =
            (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = strides[arg][i] / element_size;
      }
    }
  }

  __host__ __device__ offset_type get(index_t linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }
#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

#pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }
    }
    return offsets;
  }

  int dims;
  IntDivider<index_t> sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};

template <typename T>
std::array<int64_t, DDim::kMaxRank> ComputeStrides(
    const phi::DenseTensor& input, const size_t index_dims_size) {
  const auto& input_strides = input.strides();
  const size_t element_size_bytes = sizeof(T);

  std::array<int64_t, DDim::kMaxRank> strides{};

  for (int i = 0; i < index_dims_size; ++i) {
    if (i < input_strides.size()) {
      strides[i] = input_strides[i] * element_size_bytes;
    } else {
      strides[i] = 0;
    }
  }

  return strides;
}

template <typename IndexT>
std::array<char*, DDim::kMaxRank> GetIndexDataPtrs(
    const std::vector<const DenseTensor*> index) {
  std::array<char*, DDim::kMaxRank> index_ptrs{};

  PADDLE_ENFORCE_LE(index.size(),
                    DDim::kMaxRank,
                    "The number of index tensors exceeds the maximum rank.");

  for (size_t i = 0; i < index.size(); ++i) {
    const IndexT* p_index = index[i]->data<IndexT>();

    PADDLE_ENFORCE(p_index != nullptr,
                   "The pointer p_index is nullptr, "
                   "please check whether the index tensor is valid and "
                   "its data is correctly initialized.");

    index_ptrs[i] = reinterpret_cast<char*>(const_cast<IndexT*>(p_index));
  }

  return index_ptrs;
}

template <int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_offset_calculator(
    const DenseTensor& output,
    const DenseTensor& input,
    const std::vector<const DenseTensor*> index) {
  int ndim = output.dims().size();
  const int64_t* shape = output.dims().Get();
  std::vector<int64_t> shape_vec(shape, shape + ndim);
  std::reverse(shape_vec.begin(), shape_vec.end());
  const int64_t* desired_shape = shape_vec.data();

  std::vector<std::vector<int64_t>> strides;
  std::vector<const DenseTensor*> tensors = {&output, &input};

  for (const auto& idx_tensor : index) {
    tensors.push_back(idx_tensor);
  }

  for (const auto& tensor : tensors) {
    std::vector<int64_t> stride_bytes(ndim, 0);
    const auto& original_shape = tensor->dims();
    const auto& original_strides = tensor->strides();
    int64_t element_size_in_bytes = phi::SizeOf(tensor->dtype());
    int offset = ndim - original_shape.size();

    if (tensor == &input) {
      stride_bytes[ndim - 1] = element_size_in_bytes;
    } else {
      if (offset > 0) {
        stride_bytes.resize(ndim, 0);
      } else {
        stride_bytes.resize(ndim);
      }

      for (int i = 0; i < original_shape.size(); ++i) {
        if (original_shape[i] == 1 && shape[offset + i] != 1) {
          stride_bytes[offset + i] = 0;
        } else {
          stride_bytes[offset + i] =
              original_strides[i] * element_size_in_bytes;
        }
      }
    }
    std::reverse(stride_bytes.begin(), stride_bytes.end());
    strides.push_back(stride_bytes);
  }

  std::array<const int64_t*, N> strides_array;
  for (int i = 0; i < N; ++i) {
    strides_array[i] = strides[i].data();
  }

  return OffsetCalculator<N, uint32_t, signed_strides>(
      ndim, desired_shape, strides_array.data());
}

template <typename T, typename IndexT = int>
void IndexElementwiseKernel(const phi::GPUContext& ctx,
                            const DenseTensor& input,
                            const std::vector<const DenseTensor*> index,
                            const std::vector<int64_t>& index_dims,
                            DenseTensor* output) {
  auto num_indices = index_dims.size();

  auto index_stride = ComputeStrides<T>(input, num_indices);
  auto index_ptrs = GetIndexDataPtrs<IndexT>(index);

  auto sizes = std::array<int64_t, DDim::kMaxRank>{};
  auto strides = std::array<int64_t, DDim::kMaxRank>{};

  for (unsigned i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_stride[i];
  }

  auto offset_calc = make_offset_calculator<3>(*output, input, index);

  const int64_t N = output->numel();
  PADDLE_ENFORCE(N >= 0 && N <= std::numeric_limits<int32_t>::max(),
                 "N >= 0 && N <= std::numeric_limits<int32_t>::max()");
  constexpr int nt = launch_size_nd;
  constexpr int vt = launch_bound2;
  const dim3 block(nt);
  const dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = ctx.stream();

  using dtype = OpaqueType<sizeof(T)>;

  const char* in_ptr = reinterpret_cast<const char*>(input.data<T>());
  char* out_ptr = reinterpret_cast<char*>(output->data<T>());

  index_elementwise_kernel<nt, vt>
      <<<grid, block, 0, stream>>>(N, [=] __device__(int idx) {
        const auto offsets = offset_calc.get(idx);
        char* const out_data = out_ptr + offsets[0];
        const char* const in_data = in_ptr + offsets[1];

        int64_t offset = 0;
#pragma unroll
        for (int i = 0; i < num_indices; i++) {
          int64_t index =
              *reinterpret_cast<int64_t*>(index_ptrs[i] + offsets[2]);
          PADDLE_ENFORCE(-sizes[i] <= index && index < sizes[i],
                         "index out of bounds");
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }

        *reinterpret_cast<dtype*>(out_data) =
            *reinterpret_cast<const dtype*>(in_data + offset);
      });
}

}  // namespace funcs
}  // namespace phi
