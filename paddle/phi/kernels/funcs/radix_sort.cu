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

#include "paddle/phi/kernels/funcs/radix_sort.h"
#include "paddle/phi/common/memory_utils.h"

namespace phi {
namespace funcs {

#ifdef PADDLE_WITH_CUDA
namespace {
template <typename T>
struct CudaType {
  using type = T;
};

template <>
struct CudaType<int64_t> {
  using type = long long;  // NOLINT
};

#define PADDLE_CUB_WRAPPER(func, ...)                                     \
  do {                                                                    \
    size_t temp_storage_bytes = 0;                                        \
    func(nullptr, temp_storage_bytes, __VA_ARGS__);                       \
    auto temp_storage =                                                   \
        phi::memory_utils::Alloc(dev_ctx.GetPlace(), temp_storage_bytes); \
    func(temp_storage->ptr(), temp_storage_bytes, __VA_ARGS__);           \
  } while (0)

}  // namespace

template <typename key_t, int value_size>
void RadixSortPairsImpl(const phi::GPUContext& dev_ctx,
                        const key_t* keys_in,
                        key_t* keys_out,
                        const OpaqueTypeRadix<value_size>* values_in,
                        OpaqueTypeRadix<value_size>* values_out,
                        int64_t n,
                        bool descending,
                        int64_t begin_bit,
                        int64_t end_bit) {
  PADDLE_ENFORCE_LE(
      n,
      std::numeric_limits<int>::max(),
      phi::errors::InvalidArgument(
          "CUB sort does not support sorting more than INT_MAX elements"));

  using key_t_ = typename CudaType<key_t>::type;

  phi::Allocator::AllocationPtr keys_out_owner;
  if (keys_out == nullptr) {
    keys_out_owner =
        phi::memory_utils::Alloc(dev_ctx.GetPlace(), n * sizeof(key_t));
    keys_out = reinterpret_cast<key_t*>(keys_out_owner->ptr());
  }

  const key_t_* keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_* keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    PADDLE_CUB_WRAPPER(cub::DeviceRadixSort::SortPairsDescending,
                       keys_in_,
                       keys_out_,
                       values_in,
                       values_out,
                       static_cast<int>(n),
                       begin_bit,
                       end_bit,
                       dev_ctx.stream());
  } else {
    PADDLE_CUB_WRAPPER(cub::DeviceRadixSort::SortPairs,
                       keys_in_,
                       keys_out_,
                       values_in,
                       values_out,
                       static_cast<int>(n),
                       begin_bit,
                       end_bit,
                       dev_ctx.stream());
  }
}

#define INSTANTIATE_SORT_PAIRS(key_t, value_size)      \
  template void RadixSortPairsImpl<key_t, value_size>( \
      const phi::GPUContext&,                          \
      const key_t*,                                    \
      key_t*,                                          \
      const OpaqueTypeRadix<value_size>*,              \
      OpaqueTypeRadix<value_size>*,                    \
      int64_t,                                         \
      bool,                                            \
      int64_t,                                         \
      int64_t);

INSTANTIATE_SORT_PAIRS(int32_t, 1)
INSTANTIATE_SORT_PAIRS(int32_t, 2)
INSTANTIATE_SORT_PAIRS(int32_t, 4)
INSTANTIATE_SORT_PAIRS(int64_t, 1)
INSTANTIATE_SORT_PAIRS(int64_t, 2)
INSTANTIATE_SORT_PAIRS(int64_t, 4)
INSTANTIATE_SORT_PAIRS(int32_t, 8)
INSTANTIATE_SORT_PAIRS(int64_t, 8)

#endif
}  // namespace funcs
}  // namespace phi
