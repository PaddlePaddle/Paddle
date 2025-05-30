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
#include "paddle/phi/kernels/funcs/stride_utils.h"
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

// static inline std::vector<int64_t> infer_size_dimvector(std::vector<int64_t>
// a, std::vector<int64_t> b) {
//   // Use ptrdiff_t to ensure signed comparison.
//   auto dimsA = a.size();
//   auto dimsB = b.size();
//   auto ndim = dimsA > dimsB ? dimsA : dimsB;
//   std::vector<int64_t> expandedSizes = std::vector<int64_t> (ndim, 0);

//   for (int64_t i = ndim - 1; i >= 0; --i) {
//     int64_t offset = ndim - 1 - i;
//     int64_t dimA = dimsA - 1 - offset;
//     int64_t dimB = dimsB - 1 - offset;
//     auto sizeA = (dimA >= 0) ? a[dimA] : 1;
//     auto sizeB = (dimB >= 0) ? b[dimB] : 1;

//     expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
//   }

//   return expandedSizes;
// }

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

// static inline std::vector<int64_t> compute_strides(
//     std::vector<int64_t> input_dims, // value_tensor
//     std::vector<int64_t> input_strides,
//     int64_t input_elesize,
//     int64_t ndim,
//     std::vector<int64_t> shape_,
//     std::vector<int64_t> &stride_size
// ) {
//     std::vector<int64_t> stride_bytes(ndim, 0);
//     const auto& original_shape = input_dims;
//     const auto& original_stride = input_strides;
//     int64_t element_size_in_bytes = input_elesize;
//     int offset = ndim - original_shape.size();

//     if (offset > 0)
//         stride_bytes.resize(ndim, 0);
//     else
//         stride_bytes.resize(ndim);
//     for (int i=0; i<original_shape.size(); i++) {
//       if (original_shape[i] == 1 && shape_[offset + i] !=1) {
//         stride_bytes[offset + i] = 0;
//       } else {
//         stride_bytes[offset + i] = original_stride[i] *
//         element_size_in_bytes;
//       }
//     }
//     stride_size.push_back(stride_bytes.size());
//     return stride_bytes;
// }

// static inline std::vector<int64_t> compute_shapes(
//     std::vector<std::vector<int64_t>> input_dims
// ) {
//   std::vector<int64_t> shape_;
//   for (size_t i=0; i<input_dims.size(); i++) {
//     auto shape = input_dims[i];
//     if (shape_.empty()) {
//       shape_ = shape;
//     } else if (!(shape == shape_)) {
//       shape_ = infer_size_dimvector(shape_, shape);
//     }
//   }
//   return shape_;
// }

// template <int N>
// static inline void permute_dimensions(
//   std::array<int64_t*, N>& strides_array,
//   std::vector<int64_t>& stride_size,
//   std::vector<int64_t>& perm,
//   std::vector<int64_t>& shape_) {

//   auto reorder = [perm](std::vector<int64_t> data) {
//     auto res = std::vector<int64_t>(data.size(), 0);
//     for (int64_t i=0; i<perm.size(); i++) {
//       res[i] = data[perm[i]];
//     }
//     return res;
//   };

//   // Update shape and strides
//   shape_ = reorder(shape_);

//   static std::array<std::vector<int64_t>, N> temp_strides;
//   for (int64_t i = 0; i < N; i++) {
//     if (strides_array[i] != nullptr) {
//       std::vector<int64_t> original_data(strides_array[i], strides_array[i] +
//       stride_size[i]); temp_strides[i] = reorder(original_data);
//       strides_array[i] = temp_strides[i].data();
//     }
//   }

// }

// template <int N>
// static inline void reorder_dimensions(
//   std::vector<int64_t>& shape_,
//   std::vector<int64_t>& stride_size,
//   std::array<int64_t*, N>& strides_array) {
//   // Sort the dimensions based on strides in ascending order with reduced
//   dims
//   // at the front. NOTE: that this inverts the order of C-contiguous tensors.
//   // strides[0] is the fastest moving dimension instead of strides[ndim - 1].
//   // See NOTE: [Computing output strides] and inline  comments for more
//   detailed description

//   auto ndim = shape_.size();
//   std::vector<int64_t> perm_;

//   perm_.resize(ndim);
//   if (ndim == 1) {
//     perm_[0] = 0;
//     return;
//   }

//   // initialize perm with n-1, n-2, ..., 1, 0
//   std::iota(perm_.rbegin(), perm_.rend(), 0);

//   // Reordering dimensions changes iteraton order
//   // if (enforce_linear_iteration_) {
//   //   permute_dimensions(perm_);
//   //   return;
//   // }

//   // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
//   // before dim1, and 0 if the comparison is ambiguous.
//   auto should_swap = [&](size_t dim0, size_t dim1) {
//     for (int64_t arg=0; arg < N; arg++) {
//       // ignore undefined or incorrectly sized tensors
//       if (strides_array[arg] == nullptr) {
//         continue;
//       }
//       int64_t stride0 = strides_array[arg][dim0];
//       int64_t stride1 = strides_array[arg][dim1];
//       //move on to the next input if one of the dimensions is broadcasted
//       if (stride0 == 0 || stride1 == 0) {
//         continue;
//       // it is important to return here only with strict comparisons, for
//       equal strides we try to break the tie later
//       // by comparing corresponding dimensions or if that does not work,
//       moving on to the next tensor } else if (stride0 < stride1) {
//         return -1;
//       } else  if (stride0 > stride1) {
//         return 1;
//       } else { //equal strides, use dimensions themselves as the tie-breaker.
//         //at this point, with zero strides out of the way, we are guaranteed
//         that operand dimensions are equal to shape_
//          auto t_dim0 = shape_[dim0];
//          auto t_dim1 = shape_[dim1];
//          //return only if dimensions should be swapped, otherwise move on to
//          the next tensor if (t_dim0 > t_dim1) {
//              return 1;
//          }
//       }
//     }
//     return 0;
//   };
//   // insertion sort with support for ambiguous comparisons
//   for (int64_t i=0; i<ndim; i++) {
//     int dim1 = i;
//     for (int dim0 = i - 1; dim0 >= 0; dim0--) {
//       int comparison = should_swap(perm_[dim0], perm_[dim1]);
//       if (comparison > 0) {
//         std::swap(perm_[dim0], perm_[dim1]);
//         dim1 = dim0;
//       } else if (comparison < 0) {
//         break;
//       }
//     }
//   }

//   // perform re-ordering of shape and strides
//   permute_dimensions<N>(strides_array, stride_size, perm_, shape_);
// }

// template<int N>
// void coalesce_dimensions(
//   int64_t ndim,
//   std::array<int64_t*, N>& strides_array,
//   std::vector<int64_t> &stride_size,
//   std::vector<int64_t> &shape_
// ) {
//       for (size_t i=0; i<N; i++) {
//         int64_t* stride_tmp = strides_array[i];
//       }

//   if (ndim <= 1) {
//     return;
//   }

//   // We can coalesce two adjacent dimensions if either dim has size 1 or if:
//   // shape[n] * stride[n] == stride[n + 1].
//   auto can_coalesce = [&](int dim0, int dim1) {
//     auto shape0 = shape_[dim0];
//     auto shape1 = shape_[dim1];
//     if (shape0 == 1 || shape1 == 1) {
//       return true;
//     }
//     for (int64_t i=0; i<N; i++) {
//       auto& stride = strides_array[i];
//       if (shape0 * stride[dim0] != stride[dim1]) {
//         return false;
//       }
//     }
//     return true;
//   };

//   // replace each operands stride at dim0 with its stride at dim1
//   auto replace_stride = [&](int dim0, int dim1) {
//     for (int64_t i=0; i<N; i++) {
//       auto& stride = strides_array[i];
//       stride[dim0] = stride[dim1];
//     }
//   };

//   int prev_dim = 0;
//   for (int64_t dim=1; dim<ndim; dim++) {
//     if (can_coalesce(prev_dim, dim)) {
//       if (shape_[prev_dim] == 1) {
//         replace_stride(prev_dim, dim);
//       }
//       shape_[prev_dim] *= shape_[dim];
//     } else {
//       prev_dim++;
//       if (prev_dim != dim) {
//         replace_stride(prev_dim, dim);
//         shape_[prev_dim] = shape_[dim];
//       }
//     }
//   }
//   shape_.resize(prev_dim + 1);
//   for (int64_t i=0; i<N; i++) {
//     stride_size[i] = shape_.size();
//   }
// }

// template <int N, bool signed_strides = false>
// static OffsetCalculator<N, uint32_t, signed_strides>
// make_offset_calculator_put(
//   IndexPutStride index_put_stride
//     ) {
//   return OffsetCalculator<N, uint32_t, signed_strides>(
//       index_put_stride.desired_shape_.size(),
//       index_put_stride.desired_shape_.data(),
//       index_put_stride.strides_array_.data());
// }

// template <int N, bool signed_strides = false>
// static OffsetCalculator<N, uint32_t, signed_strides>
// make_offset_calculator_put(
//     std::vector<int64_t> output_dims, // value_tensor
//     std::vector<int64_t> output_strides,
//     int64_t output_elesize,
//     std::vector<int64_t> input_dims, // input_tensor
//     std::vector<int64_t> input_strides,
//     int64_t input_elesize,
//     std::vector<int64_t> index_dims, // index_tensor
//     std::vector<int64_t> index_strides,
//     int64_t index_elesize,
//     int64_t &numel
//     ) {
//   int ndim = output_dims.size();
//   // need a 2D stride vector to hold stride for each
//   std::array<int64_t*, N> strides_array;
//   std::array<std::vector<int64_t>, N> strides_vec;
//   std::vector<int64_t> stride_size;

//   std::vector<int64_t> desired_shape = compute_shapes(
//     {input_dims, output_dims, index_dims}
//   );

//   // dangling pointer
//   strides_vec[0] = compute_strides(
//     output_dims, // input_tensor
//     output_strides,
//     output_elesize,
//     ndim,
//     desired_shape,
//     stride_size
//   );

//   strides_vec[1] = compute_strides(
//     input_dims, // value_tensor
//     input_strides,
//     input_elesize,
//     ndim,
//     desired_shape,
//     stride_size
//   );

//   strides_vec[2] = compute_strides(
//     index_dims, // index_tensor
//     index_strides,
//     index_elesize,
//     ndim,
//     desired_shape,
//     stride_size
//   );

//       for (size_t i=0; i<N; i++) {
//         strides_array[i] = strides_vec[i].data();
//       }

//   reorder_dimensions<N>(desired_shape, stride_size, strides_array);

//   coalesce_dimensions<N>(ndim, strides_array, stride_size, desired_shape);

//   int num = 1;
//   for (int i=0; i<desired_shape.size(); i++) {
//     num *= desired_shape[i];
//   }
//   numel = num;

//   return OffsetCalculator<N, uint32_t, signed_strides>(
//       desired_shape.size(), desired_shape.data(), strides_array.data());
// }

template <int N, bool signed_strides = false>
static OffsetCalculator<N, uint32_t, signed_strides> make_offset_calculator_put(
    std::vector<int64_t> desired_shape, std::array<int64_t*, N> strides_array) {
  // int ndim = output_dims.size();
  // // need a 2D stride vector to hold stride for each
  // std::array<int64_t*, N> strides_array;
  // std::array<std::vector<int64_t>, N> strides_vec;
  // std::vector<int64_t> stride_size;

  // std::vector<int64_t> desired_shape = compute_shapes(
  //   {input_dims, output_dims, index_dims}
  // );

  // // dangling pointer
  // strides_vec[0] = compute_strides(
  //   output_dims, // input_tensor
  //   output_strides,
  //   output_elesize,
  //   ndim,
  //   desired_shape,
  //   stride_size
  // );

  // strides_vec[1] = compute_strides(
  //   input_dims, // value_tensor
  //   input_strides,
  //   input_elesize,
  //   ndim,
  //   desired_shape,
  //   stride_size
  // );

  // strides_vec[2] = compute_strides(
  //   index_dims, // index_tensor
  //   index_strides,
  //   index_elesize,
  //   ndim,
  //   desired_shape,
  //   stride_size
  // );

  //     for (size_t i=0; i<N; i++) {
  //       strides_array[i] = strides_vec[i].data();
  //     }

  // reorder_dimensions<N>(desired_shape, stride_size, strides_array);

  // coalesce_dimensions<N>(ndim, strides_array, stride_size, desired_shape);

  // int num = 1;
  // for (int i=0; i<desired_shape.size(); i++) {
  //   num *= desired_shape[i];
  // }
  // numel = num;

  return OffsetCalculator<N, uint32_t, signed_strides>(
      desired_shape.size(), desired_shape.data(), strides_array.data());
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

}  // namespace funcs
}  // namespace phi
