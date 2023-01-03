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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/fast_divmod.h"

namespace phi {
namespace funcs {

template <typename IndexT>
struct DivmodWarpper {
 public:
  explicit DivmodWarpper(IndexT d) { divmoder = phi::funcs::FastDivMod(d); }
  __device__ inline phi::funcs::FastDivMod::DivModT div_mod(IndexT val) {
    return divmoder.Divmod(val);
  }

 private:
  phi::funcs::FastDivMod divmoder;
};

template <>
struct DivmodWarpper<int64_t> {
 public:
  using DivModT = phi::AlignedVector<int64_t, 2>;

  explicit DivmodWarpper(int64_t d) { divisor = d; }
  __device__ inline DivModT div_mod(int64_t val) {
    DivModT data;
    data[0] = val / divisor;
    data[1] = val - data[0] * divisor;
    return data;
  }

 private:
  int64_t divisor;
};

enum class SegmentedArraySize {
  kVariableLength = 0,
  kFixed16 = 16,
  kFixed32 = 32,
  kFixed64 = 64,
};

template <typename T, SegmentedArraySize Size>
struct ConstPointerArray {
 public:
  const T* data[static_cast<int>(Size)];

  void Set(const std::vector<const T*>& x_ptrs, const T** dev_ptr = nullptr) {
    for (auto i = 0; i < x_ptrs.size(); ++i) {
      data[i] = x_ptrs[i];
    }
  }
};

template <typename T>
struct ConstPointerArray<T, SegmentedArraySize::kVariableLength> {
 public:
  const T** data{nullptr};

  void Set(const std::vector<const T*>& x_ptrs, const T** dev_ptr = nullptr) {
    data = dev_ptr;
  }
};

template <typename T, SegmentedArraySize Size>
struct PointerArray {
 public:
  T* data[static_cast<int>(Size)];

  void Set(const std::vector<T*>& x_ptrs, T** dev_ptr = nullptr) {
    for (auto i = 0; i < x_ptrs.size(); ++i) {
      data[i] = x_ptrs[i];
    }
  }
};

template <typename T>
struct PointerArray<T, SegmentedArraySize::kVariableLength> {
 public:
  T** data{nullptr};

  void Set(const std::vector<T*>& x_ptrs, T** dev_ptr = nullptr) {
    data = dev_ptr;
  }
};

template <typename Context>
struct ArraySetterBase {
 protected:
  void* AllocAndCopy(const Context& ctx, void* src, size_t num_bytes) {
    allocation = paddle::memory::Alloc(
        ctx.GetPlace(),
        num_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    paddle::memory::Copy(ctx.GetPlace(),
                         allocation->ptr(),
                         phi::CPUPlace(),
                         src,
                         num_bytes,
                         ctx.stream());
    return allocation->ptr();
  }

  phi::Allocator::AllocationPtr allocation{nullptr};
};

template <typename Context,
          typename T,
          SegmentedArraySize Size,
          bool IsConst = true>
struct PointerArraySetter : public ArraySetterBase<Context> {
 public:
  ConstPointerArray<T, Size> array;

  PointerArraySetter(const Context& ctx,
                     const std::vector<const DenseTensor*>& x) {
    x_ptrs.resize(x.size());
    for (int i = 0; i < x.size(); ++i) {
      x_ptrs[i] = x[i]->data<T>();
    }

    const T** dev_ptr = nullptr;
    if (Size == SegmentedArraySize::kVariableLength) {
      size_t num_bytes = x.size() * sizeof(T*);
      dev_ptr = reinterpret_cast<const T**>(
          AllocAndCopy(ctx, reinterpret_cast<void*>(x_ptrs.data()), num_bytes));
    }

    array.Set(x_ptrs, dev_ptr);
  }

 private:
  std::vector<const T*> x_ptrs;
};

inline SegmentedArraySize CalcArraySize(int n) {
  if (n <= 16) {
    return SegmentedArraySize::kFixed16;
  } else if (n <= 32) {
    return SegmentedArraySize::kFixed32;
  } else if (n <= 64) {
    return SegmentedArraySize::kFixed64;
  } else {
    return SegmentedArraySize::kVariableLength;
  }
}
}  // namespace funcs

#define _POINTER_ARRAY_KERNEL_CASE(size, ...)                         \
  case (size): {                                                      \
    constexpr auto kArraySize = (size);                               \
    funcs::PointerArraySetter<Context, T, kArraySize> setter(ctx, x); \
    __VA_ARGS__;                                                      \
  } break

#define _POINTER_ARRAY_KERNEL_DEFAULT(size, ...)                      \
  default: {                                                          \
    constexpr auto kArraySize = (size);                               \
    funcs::PointerArraySetter<Context, T, kArraySize> setter(ctx, x); \
    __VA_ARGS__;                                                      \
  } break

#define POINTER_ARRAY_KERNEL_HELPER(...)                                    \
  _POINTER_ARRAY_KERNEL_CASE(funcs::SegmentedArraySize::kFixed16,           \
                             ##__VA_ARGS__);                                \
  _POINTER_ARRAY_KERNEL_CASE(funcs::SegmentedArraySize::kFixed32,           \
                             ##__VA_ARGS__);                                \
  _POINTER_ARRAY_KERNEL_CASE(funcs::SegmentedArraySize::kFixed64,           \
                             ##__VA_ARGS__);                                \
  _POINTER_ARRAY_KERNEL_DEFAULT(funcs::SegmentedArraySize::kVariableLength, \
                                ##__VA_ARGS__);

}  // namespace phi
