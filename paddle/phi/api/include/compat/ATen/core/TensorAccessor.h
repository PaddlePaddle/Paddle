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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <torch/headeronly/core/TensorAccessor.h>

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace at {

using torch::headeronly::DefaultPtrTraits;
#if defined(__CUDACC__) || defined(__HIPCC__)
using torch::headeronly::RestrictPtrTraits;
#endif

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
using TensorAccessorBase = torch::headeronly::detail::
    TensorAccessorBase<c10::IntArrayRef, T, N, PtrTraits, index_t>;

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
using TensorAccessor = torch::headeronly::detail::
    TensorAccessor<c10::IntArrayRef, T, N, PtrTraits, index_t>;

namespace detail {

template <size_t N, typename index_t>
struct IndexBoundsCheck {
  explicit IndexBoundsCheck(index_t i) {
    TORCH_CHECK(0 <= i && i < index_t{N},
                "Index ",
                i,
                " is not within bounds of a tensor of dimension ",
                N);
  }
};

}  // namespace detail

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
using GenericPackedTensorAccessorBase =
    torch::headeronly::detail::GenericPackedTensorAccessorBase<
        detail::IndexBoundsCheck<N, index_t>,
        T,
        N,
        PtrTraits,
        index_t>;

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits,
          typename index_t = int64_t>
using GenericPackedTensorAccessor =
    torch::headeronly::detail::GenericPackedTensorAccessor<
        TensorAccessor<T, N - 1, PtrTraits, index_t>,
        detail::IndexBoundsCheck<N, index_t>,
        T,
        N,
        PtrTraits,
        index_t>;

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor32 =
    GenericPackedTensorAccessor<T, N, PtrTraits, int32_t>;

template <typename T,
          size_t N,
          template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor64 =
    GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>;

}  // namespace at
