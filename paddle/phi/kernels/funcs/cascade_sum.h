// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"

// Bit-exact port of the single-threaded CPU cascade summation used by torch,
// see pytorch/aten/src/ATen/native/cpu/SumKernel.cpp. Torch does not sum
// sequentially: it keeps 4 ILP accumulators x 4 cascade levels so that only
// values of similar magnitude are added together. Matching torch bitwise
// therefore requires reproducing the exact accumulation order, which includes
// the TensorIterator dimension ordering and the vectorized/scalar path
// selection, not just the cascade itself.
//
// SIMD lanes are modelled as plain arrays: lane-wise scalar additions give the
// same IEEE result as the real vector instructions, so no intrinsics are
// needed.
//
// Integral types are deliberately not handled here: torch reduces them with
// `binary_kernel_reduce_vec`, and since integer addition is associative any
// summation order yields identical results already.
namespace phi {
namespace funcs {
namespace cascade_sum {

// torch::at::native::utils::CeilLog2
inline int64_t CeilLog2(int64_t x) {
  if (x <= 2) return 1;
  uint64_t v = static_cast<uint64_t>(x) - 1;
  int64_t bits = 0;
  while (v > 0) {
    v >>= 1;
    ++bits;
  }
  return bits;
}

// Byte width of Vectorized<T>. sum_stub is registered with REGISTER_DISPATCH,
// which maps the AVX512 slot to nullptr (pytorch aten/src/ATen/native/
// DispatchStub.h), so the sum kernel always runs its AVX2 / default build and
// both use 32 byte vectors.
constexpr int64_t kVecBytes = 32;

// at::acc_type<T, /*is_cuda=*/true>: reduced floating point accumulates in
// float, everything else accumulates in its own type.
template <typename T>
struct AccTypeOf {
  using type = T;
};
template <>
struct AccTypeOf<::phi::dtype::float16> {
  using type = float;
};
template <>
struct AccTypeOf<::phi::dtype::bfloat16> {
  using type = float;
};

template <typename T>
struct Traits {
  using Acc = typename AccTypeOf<T>::type;
  // Vectorized<T>::size() and Vectorized<acc_t>::size().
  static constexpr int kVecElems = static_cast<int>(kVecBytes / sizeof(T));
  static constexpr int kAccLanes = static_cast<int>(kVecBytes / sizeof(Acc));
  static constexpr int kFold = kVecElems / kAccLanes;
};

template <typename A, int N>
struct Lanes {
  A v[N];

  static Lanes Zero() {
    Lanes r;
    for (int i = 0; i < N; ++i) r.v[i] = A(0);
    return r;
  }

  Lanes& operator+=(const Lanes& o) {
    for (int i = 0; i < N; ++i) v[i] = v[i] + o.v[i];
    return *this;
  }
};

// CastLoadPolicy: one element, converted to the accumulate type.
template <typename T, typename A>
struct ScalarLoader {
  using Value = Lanes<A, 1>;
  static constexpr int64_t MemSize() { return sizeof(T); }
  static Value Load(const char* base, int64_t stride, int64_t index) {
    Value r;
    r.v[0] = static_cast<A>(*reinterpret_cast<const T*>(base + index * stride));
    return r;
  }
};

// InnerSumCastLoadPolicy: reads a full Vectorized<T> and folds it down to
// Vectorized<acc_t> lanes. For fp16/bf16 that means lane i accumulates
// x[i] + x[i + kAccLanes] before the cascade even starts (torch's
// `load_to_float` + `first + second`).
template <typename T, typename A, int VecElems, int AccLanes>
struct InnerVecLoader {
  using Value = Lanes<A, AccLanes>;
  static constexpr int64_t MemSize() { return sizeof(T) * VecElems; }
  static Value Load(const char* base, int64_t stride, int64_t index) {
    const T* p = reinterpret_cast<const T*>(base + index * stride);
    Value r;
    for (int i = 0; i < AccLanes; ++i) r.v[i] = static_cast<A>(p[i]);
    for (int k = 1; k < VecElems / AccLanes; ++k) {
      for (int i = 0; i < AccLanes; ++i) {
        r.v[i] = r.v[i] + static_cast<A>(p[k * AccLanes + i]);
      }
    }
    return r;
  }
};

// OuterSumCastLoadPolicy: reads only Vectorized<acc_t>::size() elements.
template <typename T, typename A, int AccLanes>
struct OuterVecLoader {
  using Value = Lanes<A, AccLanes>;
  static constexpr int64_t MemSize() { return sizeof(T) * AccLanes; }
  static Value Load(const char* base, int64_t stride, int64_t index) {
    const T* p = reinterpret_cast<const T*>(base + index * stride);
    Value r;
    for (int i = 0; i < AccLanes; ++i) r.v[i] = static_cast<A>(p[i]);
    return r;
  }
};

// CastStoreAccumulate: `*ptr += value` with ptr being T* and value acc_t. For
// fp16/bf16 the only viable overload is `operator+=(Half&, const Half&)`, so
// torch rounds the incoming accumulator to T *before* adding and rounds again
// on assignment. Locations written more than once (out_stride == 0, or several
// 2d loop calls hitting the same element) depend on both roundings.
template <typename T, typename A>
inline void StoreAdd(char* data, A value) {
  T* p = reinterpret_cast<T*>(data);
  const A rhs = static_cast<A>(static_cast<T>(value));
  *p = static_cast<T>(static_cast<A>(*p) + rhs);
}

template <typename Loader, int64_t nrows>
std::array<typename Loader::Value, nrows> MultiRowSum(const char* in_data,
                                                      int64_t row_stride,
                                                      int64_t col_stride,
                                                      int64_t size) {
  using AccT = typename Loader::Value;
  constexpr int64_t num_levels = 4;

  const int64_t level_power = std::max<int64_t>(4, CeilLog2(size) / num_levels);
  const int64_t level_step = (int64_t{1} << level_power);
  const int64_t level_mask = level_step - 1;

  std::array<std::array<AccT, nrows>, num_levels> acc;
  for (auto& row : acc) {
    for (auto& a : row) a = AccT::Zero();
  }

  int64_t i = 0;
  for (; i + level_step <= size;) {
    for (int64_t j = 0; j < level_step; ++j, ++i) {
      const char* sum_base = in_data + i * row_stride;
      for (int64_t k = 0; k < nrows; ++k) {
        acc[0][k] += Loader::Load(sum_base, col_stride, k);
      }
    }

    for (int64_t j = 1; j < num_levels; ++j) {
      for (int64_t k = 0; k < nrows; ++k) {
        acc[j][k] += acc[j - 1][k];
        acc[j - 1][k] = AccT::Zero();
      }
      const int64_t mask = (level_mask << (j * level_power));
      if ((i & mask) != 0) break;
    }
  }

  for (; i < size; ++i) {
    const char* sum_base = in_data + i * row_stride;
    for (int64_t k = 0; k < nrows; ++k) {
      acc[0][k] += Loader::Load(sum_base, col_stride, k);
    }
  }

  for (int64_t j = 1; j < num_levels; ++j) {
    for (int64_t k = 0; k < nrows; ++k) {
      acc[0][k] += acc[j][k];
    }
  }
  return acc[0];
}

template <typename Loader>
typename Loader::Value RowSum(const char* in_data,
                              int64_t in_stride,
                              int64_t size) {
  constexpr int64_t ilp_factor = 4;

  const int64_t size_ilp = size / ilp_factor;
  auto partial_sums = MultiRowSum<Loader, ilp_factor>(
      in_data, in_stride * ilp_factor, in_stride, size_ilp);

  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_sums[0] += Loader::Load(in_data, in_stride, i);
  }

  for (int64_t k = 1; k < ilp_factor; ++k) {
    partial_sums[0] += partial_sums[k];
  }
  return partial_sums[0];
}

template <typename T>
void VectorizedInnerSum(const char* in_data,
                        char* out_data,
                        int64_t outer_stride,
                        int64_t out_stride,
                        int64_t size0,
                        int64_t size1) {
  using Tr = Traits<T>;
  using A = typename Tr::Acc;
  using VecLoader = InnerVecLoader<T, A, Tr::kVecElems, Tr::kAccLanes>;
  using ScaLoader = ScalarLoader<T, A>;

  const int64_t vec_size = size0 / Tr::kVecElems;

  for (int64_t j = 0; j < size1; ++j) {
    const char* row_in = in_data + j * outer_stride;
    auto vec_acc = RowSum<VecLoader>(row_in, VecLoader::MemSize(), vec_size);

    A final_acc = A(0);
    for (int64_t k = vec_size * Tr::kVecElems; k < size0; ++k) {
      final_acc = final_acc + ScaLoader::Load(row_in, sizeof(T), k).v[0];
    }
    for (int i = 0; i < Tr::kAccLanes; ++i) {
      final_acc = final_acc + vec_acc.v[i];
    }
    StoreAdd<T, A>(out_data + j * out_stride, final_acc);
  }
}

template <typename T>
void ScalarInnerSum(const char* in_data,
                    char* out_data,
                    const int64_t in_strides[2],
                    int64_t out_stride,
                    int64_t size0,
                    int64_t size1) {
  using A = typename Traits<T>::Acc;
  using ScaLoader = ScalarLoader<T, A>;
  for (int64_t j = 0; j < size1; ++j) {
    const char* row_in = in_data + j * in_strides[1];
    auto ans = RowSum<ScaLoader>(row_in, in_strides[0], size0);
    StoreAdd<T, A>(out_data + j * out_stride, ans.v[0]);
  }
}

template <typename T>
void VectorizedOuterSum(const char* in_data,
                        char* out_data,
                        int64_t inner_stride,
                        int64_t out_stride,
                        int64_t size0,
                        int64_t size1) {
  using Tr = Traits<T>;
  using A = typename Tr::Acc;
  using VecLoader = OuterVecLoader<T, A, Tr::kAccLanes>;
  using ScaLoader = ScalarLoader<T, A>;
  constexpr int64_t nrows = 4;
  constexpr int64_t lanes = Tr::kAccLanes;

  int64_t j = 0;
  for (; j + nrows * lanes <= size1; j += nrows * lanes) {
    const char* row_in = in_data + j * sizeof(T);
    auto sums = MultiRowSum<VecLoader, nrows>(
        row_in, inner_stride, VecLoader::MemSize(), size0);
    for (int64_t i = 0; i < nrows; ++i) {
      char* base = out_data + out_stride * (j + i * lanes);
      for (int64_t k = 0; k < lanes; ++k) {
        StoreAdd<T, A>(base + k * out_stride, sums[i].v[k]);
      }
    }
  }

  for (; j + lanes <= size1; j += lanes) {
    const char* row_in = in_data + j * sizeof(T);
    auto sums = RowSum<VecLoader>(row_in, inner_stride, size0);
    char* base = out_data + out_stride * j;
    for (int64_t k = 0; k < lanes; ++k) {
      StoreAdd<T, A>(base + k * out_stride, sums.v[k]);
    }
  }

  for (; j < size1; ++j) {
    const char* row_in = in_data + j * sizeof(T);
    auto ans = RowSum<ScaLoader>(row_in, inner_stride, size0);
    StoreAdd<T, A>(out_data + j * out_stride, ans.v[0]);
  }
}

template <typename T>
void ScalarOuterSum(const char* in_data,
                    char* out_data,
                    const int64_t in_strides[2],
                    int64_t out_stride,
                    int64_t size0,
                    int64_t size1) {
  using A = typename Traits<T>::Acc;
  using ScaLoader = ScalarLoader<T, A>;
  constexpr int64_t nrows = 4;

  int64_t j = 0;
  for (; j + (nrows - 1) < size1; j += nrows) {
    const char* row_in = in_data + j * in_strides[1];
    auto sums = MultiRowSum<ScaLoader, nrows>(
        row_in, in_strides[0], in_strides[1], size0);
    char* base = out_data + out_stride * j;
    for (int64_t k = 0; k < nrows; ++k) {
      StoreAdd<T, A>(base + k * out_stride, sums[k].v[0]);
    }
  }

  for (; j < size1; ++j) {
    const char* row_in = in_data + j * in_strides[1];
    auto ans = RowSum<ScaLoader>(row_in, in_strides[0], size0);
    StoreAdd<T, A>(out_data + j * out_stride, ans.v[0]);
  }
}

// Reached when neither loop dimension is reduced, which can only happen once
// every reduced dimension has been coalesced away into a kept one. Torch
// degenerates to a plain elementwise `out += in` here.
template <typename T>
void NotAReduction(const char* in_data,
                   char* out_data,
                   const int64_t in_strides[2],
                   const int64_t out_strides[2],
                   int64_t size0,
                   int64_t size1) {
  using A = typename Traits<T>::Acc;
  for (int64_t o = 0; o < size1; ++o) {
    char* out_row = out_data + o * out_strides[1];
    const char* in_row = in_data + o * in_strides[1];
    for (int64_t i = 0; i < size0; ++i) {
      const A value = static_cast<A>(
          *reinterpret_cast<const T*>(in_row + i * in_strides[0]));
      StoreAdd<T, A>(out_row + i * out_strides[0], value);
    }
  }
}

// The body of torch's `cascade_sum` 2d loop. All strides are in bytes.
template <typename T>
void Loop2d(const char* in_data,
            char* out_data,
            const int64_t in_strides_in[2],
            const int64_t out_strides_in[2],
            int64_t size0,
            int64_t size1) {
  int64_t in_strides[2] = {in_strides_in[0], in_strides_in[1]};
  int64_t out_strides[2] = {out_strides_in[0], out_strides_in[1]};

  if (out_strides[0] != 0 && out_strides[1] == 0) {
    std::swap(in_strides[0], in_strides[1]);
    std::swap(out_strides[0], out_strides[1]);
    std::swap(size0, size1);
  }

  if (out_strides[0] != 0 && out_strides[1] != 0) {
    NotAReduction<T>(
        in_data, out_data, in_strides_in, out_strides_in, size0, size1);
    return;
  }

  const int64_t out_stride = out_strides[1];
  constexpr int64_t kElemSize = static_cast<int64_t>(sizeof(T));
  constexpr int64_t kVecElems = Traits<T>::kVecElems;

  if (in_strides[0] == kElemSize && size0 >= kVecElems) {
    VectorizedInnerSum<T>(
        in_data, out_data, in_strides[1], out_stride, size0, size1);
  } else if (in_strides[1] == kElemSize && size1 >= kVecElems) {
    VectorizedOuterSum<T>(
        in_data, out_data, in_strides[0], out_stride, size0, size1);
  } else if (in_strides[0] < in_strides[1]) {
    ScalarInnerSum<T>(in_data, out_data, in_strides, out_stride, size0, size1);
  } else {
    ScalarOuterSum<T>(in_data, out_data, in_strides, out_stride, size0, size1);
  }
}

struct DimInfo {
  int64_t size;
  int64_t in_stride;   // bytes
  int64_t out_stride;  // bytes, 0 for reduced dims
};

// TensorIteratorBase::reorder_dimensions()'s `should_swap`: operand 0 is the
// reduction output, operand 1 the input. Returns 1 if `a` should come after
// `b`, -1 if before, 0 if ambiguous.
inline int ShouldSwap(const DimInfo& a, const DimInfo& b) {
  for (int arg = 0; arg < 2; ++arg) {
    const int64_t stride0 = (arg == 0) ? a.out_stride : a.in_stride;
    const int64_t stride1 = (arg == 0) ? b.out_stride : b.in_stride;
    if (arg == 0) {
      // Move reduced dimensions to the front.
      if ((stride0 == 0) != (stride1 == 0)) {
        return stride1 == 0 ? 1 : -1;
      }
    }
    if (stride0 == 0 || stride1 == 0) {
      continue;
    } else if (stride0 < stride1) {
      return -1;
    } else if (stride0 > stride1) {
      return 1;
    } else if (a.size > b.size) {
      return 1;
    }
  }
  return 0;
}

// Reproduce reorder_dimensions() + coalesce_dimensions() for a reduction:
// dimensions start out reversed (fastest moving first), get insertion-sorted
// with the ambiguity-tolerant comparator above, then adjacent dimensions are
// merged when their strides allow it.
inline std::vector<DimInfo> BuildDims(const std::vector<int64_t>& shape,
                                      const std::vector<int64_t>& in_strides,
                                      const std::vector<int64_t>& out_strides) {
  const int rank = static_cast<int>(shape.size());
  std::vector<DimInfo> dims;
  dims.reserve(rank);
  for (int i = rank - 1; i >= 0; --i) {
    dims.push_back(DimInfo{shape[i], in_strides[i], out_strides[i]});
  }
  if (dims.empty()) {
    return {DimInfo{1, 0, 0}};
  }

  for (int i = 1; i < rank; ++i) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; --dim0) {
      const int comparison = ShouldSwap(dims[dim0], dims[dim1]);
      if (comparison > 0) {
        std::swap(dims[dim0], dims[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }

  std::vector<DimInfo> coalesced{dims[0]};
  for (size_t i = 1; i < dims.size(); ++i) {
    DimInfo& prev = coalesced.back();
    const DimInfo& cur = dims[i];
    const bool can_coalesce = prev.size == 1 || cur.size == 1 ||
                              (prev.size * prev.in_stride == cur.in_stride &&
                               prev.size * prev.out_stride == cur.out_stride);
    if (can_coalesce) {
      if (prev.size == 1) {
        prev.in_stride = cur.in_stride;
        prev.out_stride = cur.out_stride;
      }
      prev.size *= cur.size;
    } else {
      coalesced.push_back(cur);
    }
  }
  return coalesced;
}

// Zero-initializes `out_data` and accumulates the reduction into it, following
// torch's serial_for_each: dims[0]/dims[1] form the 2d loop, the remaining
// dimensions are iterated with dims[2] moving fastest.
template <typename T>
void Run(const T* x_data,
         T* out_data,
         int64_t out_numel,
         const std::vector<DimInfo>& dims) {
  std::memset(out_data, 0, out_numel * sizeof(T));

  const int64_t in_strides[2] = {dims[0].in_stride,
                                 dims.size() > 1 ? dims[1].in_stride : 0};
  const int64_t out_strides[2] = {dims[0].out_stride,
                                  dims.size() > 1 ? dims[1].out_stride : 0};
  const int64_t size0 = dims[0].size;
  const int64_t size1 = dims.size() > 1 ? dims[1].size : 1;

  int64_t outer_numel = 1;
  for (size_t i = 2; i < dims.size(); ++i) outer_numel *= dims[i].size;

  const char* in_base = reinterpret_cast<const char*>(x_data);
  char* out_base = reinterpret_cast<char*>(out_data);

  for (int64_t linear = 0; linear < outer_numel; ++linear) {
    int64_t in_offset = 0, out_offset = 0, rest = linear;
    for (size_t i = 2; i < dims.size(); ++i) {
      const int64_t idx = rest % dims[i].size;
      rest /= dims[i].size;
      in_offset += idx * dims[i].in_stride;
      out_offset += idx * dims[i].out_stride;
    }
    Loop2d<T>(in_base + in_offset,
              out_base + out_offset,
              in_strides,
              out_strides,
              size0,
              size1);
  }
}

// c10::TensorImpl::is_non_overlapping_and_dense(): `Tensor::to(dtype)` keeps
// the original strides for such tensors and falls back to a contiguous copy
// otherwise.
inline bool IsNonOverlappingAndDense(const std::vector<int64_t>& shape,
                                     const std::vector<int64_t>& strides) {
  std::vector<std::pair<int64_t, int64_t>> sorted;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 1) sorted.emplace_back(strides[i], shape[i]);
  }
  std::sort(sorted.begin(), sorted.end());
  int64_t expected = 1;
  for (const auto& [stride, size] : sorted) {
    if (stride != expected) return false;
    expected *= size;
  }
  return true;
}

inline std::vector<DimInfo> MakeReduceDims(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& x_strides,
    const std::vector<int64_t>& reduce_axes,
    int64_t elem_size) {
  const int rank = static_cast<int>(x_shape.size());
  std::vector<bool> is_reduced(rank, false);
  for (int64_t axis : reduce_axes) {
    is_reduced[axis] = true;
  }

  std::vector<int64_t> in_bytes(rank), out_bytes(rank, 0);
  int64_t acc_stride = 1;
  for (int i = rank - 1; i >= 0; --i) {
    in_bytes[i] = x_strides[i] * elem_size;
    if (!is_reduced[i]) {
      out_bytes[i] = acc_stride * elem_size;
      acc_stride *= x_shape[i];
    }
  }
  return BuildDims(x_shape, in_bytes, out_bytes);
}

// torch::native::should_use_acc_buffer(): both dimensions of the 2d loop are
// reduced, so fp16/bf16 partial sums would have to travel through the low
// precision output.
inline bool NeedsAccBuffer(const std::vector<DimInfo>& dims) {
  return dims.size() >= 2 && dims[0].out_stride == 0 && dims[1].out_stride == 0;
}

}  // namespace cascade_sum

// Whether torch bypasses the fp16/bf16 kernel entirely and sums a float32 copy
// of the input, rounding the result only once (pytorch issue 83149). The
// predicate does not depend on the element size.
inline bool NeedsFloatAccBuffer(const std::vector<int64_t>& x_shape,
                                const std::vector<int64_t>& x_strides,
                                const std::vector<int64_t>& reduce_axes) {
  return cascade_sum::NeedsAccBuffer(
      cascade_sum::MakeReduceDims(x_shape, x_strides, reduce_axes, 1));
}

// Strides that `Tensor::to(dtype)` produces for this layout: MemoryFormat
// ::Preserve keeps them for a non-overlapping-and-dense view and compacts
// everything else to row-major.
inline std::vector<int64_t> CastTargetStrides(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& x_strides) {
  if (cascade_sum::IsNonOverlappingAndDense(x_shape, x_strides)) {
    return x_strides;
  }
  const int rank = static_cast<int>(x_shape.size());
  std::vector<int64_t> strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * x_shape[i + 1];
  }
  return strides;
}

// Reproduces `Tensor::to(dtype)` with MemoryFormat::Preserve (torch's
// _to_copy): a non-overlapping-and-dense view keeps its exact strides, so a
// transposed view stays transposed and the reduction still sees the original
// dimension order; anything else is compacted to row-major, matching
// `empty_like(self, suggest_memory_format())`.
template <typename Src, typename Dst>
void CastPreservingLayout(const Src* x_data,
                          const std::vector<int64_t>& x_shape,
                          const std::vector<int64_t>& x_strides,
                          std::vector<Dst>* buffer,
                          std::vector<int64_t>* buffer_strides) {
  const int rank = static_cast<int>(x_shape.size());
  *buffer_strides = CastTargetStrides(x_shape, x_strides);
  if (*buffer_strides == x_strides) {
    // A dense view covers [0, extent) exactly once, so the conversion is a flat
    // copy that keeps every stride, including the contiguous case.
    int64_t extent = 1;
    for (int i = 0; i < rank; ++i) {
      extent += (x_shape[i] - 1) * x_strides[i];
    }
    buffer->resize(extent);
    for (int64_t k = 0; k < extent; ++k) {
      (*buffer)[k] = static_cast<Dst>(x_data[k]);
    }
    return;
  }

  int64_t numel = 1;
  for (int64_t size : x_shape) numel *= size;
  buffer->resize(numel);
  std::vector<int64_t> index(rank, 0);
  for (int64_t k = 0; k < numel; ++k) {
    int64_t src = 0;
    for (int i = 0; i < rank; ++i) {
      src += index[i] * x_strides[i];
    }
    (*buffer)[k] = static_cast<Dst>(x_data[src]);
    for (int i = rank - 1; i >= 0; --i) {
      if (++index[i] < x_shape[i]) break;
      index[i] = 0;
    }
  }
}

// Torch-compatible reduce sum. `x_strides` are element strides of the (possibly
// non-contiguous) input; `reduce_axes` must be non-negative and distinct. The
// output is assumed contiguous over the kept axes.
template <typename T>
void TorchCompatibleReduceSum(const T* x_data,
                              const std::vector<int64_t>& x_shape,
                              const std::vector<int64_t>& x_strides,
                              const std::vector<int64_t>& reduce_axes,
                              T* out_data,
                              int64_t out_numel) {
  auto dims = cascade_sum::MakeReduceDims(
      x_shape, x_strides, reduce_axes, static_cast<int64_t>(sizeof(T)));

  using Acc = typename cascade_sum::AccTypeOf<T>::type;
  if constexpr (!std::is_same_v<T, Acc>) {
    if (cascade_sum::NeedsAccBuffer(dims)) {
      // `x_data` is already in T here, so this only reproduces torch for a
      // same-dtype reduction; the dtype promoting case is handled by the
      // caller.
      std::vector<Acc> acc_buffer;
      std::vector<int64_t> acc_strides;
      CastPreservingLayout<T, Acc>(
          x_data, x_shape, x_strides, &acc_buffer, &acc_strides);

      std::vector<Acc> acc_out(out_numel);
      TorchCompatibleReduceSum<Acc>(acc_buffer.data(),
                                    x_shape,
                                    acc_strides,
                                    reduce_axes,
                                    acc_out.data(),
                                    out_numel);
      for (int64_t i = 0; i < out_numel; ++i) {
        out_data[i] = static_cast<T>(acc_out[i]);
      }
      return;
    }
  }

  cascade_sum::Run<T>(x_data, out_data, out_numel, dims);
}

// An empty `dims` or `reduce_all` means every axis is reduced, matching
// recompute_reduce_all().
inline std::vector<int64_t> NormalizeReduceAxes(
    const DDim& x_dims, const std::vector<int64_t>& dims, bool reduce_all) {
  const int rank = x_dims.size();
  std::vector<int64_t> axes;
  if (reduce_all || dims.empty() || static_cast<int>(dims.size()) == rank) {
    axes.reserve(rank);
    for (int i = 0; i < rank; ++i) axes.push_back(i);
    return axes;
  }
  axes.reserve(dims.size());
  for (int64_t axis : dims) {
    axes.push_back(axis < 0 ? axis + rank : axis);
  }
  return axes;
}

}  // namespace funcs
}  // namespace phi
