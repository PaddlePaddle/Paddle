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

#include "paddle/fluid/memory/memory.h"

namespace phi {
namespace funcs {

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

  void Set(const std::vector<const T*>& x_ptrs, void* dev_ptr = nullptr) {
    for (auto i = 0; i < x_ptrs.size(); ++i) {
      data[i] = x_ptrs[i];
    }
  }
};

template <typename T>
struct ConstPointerArray<T, SegmentedArraySize::kVariableLength> {
 public:
  T** data{nullptr};

  void Set(const std::vector<const T*>& x_ptrs, void* dev_ptr = nullptr) {
    data = reinterpret_cast<T**>(dev_ptr);
  }
};

template <typename Context, typename T, SegmentedArraySize Size>
struct PointerArraySetter {
 public:
  ConstPointerArray<T, Size> array;

  PointerArraySetter(const Context& ctx,
                     const std::vector<const DenseTensor*>& x) {
    x_ptrs.resize(x.size());
    for (int i = 0; i < x.size(); ++i) {
      x_ptrs[i] = x[i]->data<T>();
    }

    void* dev_ptr = nullptr;
    if (Size == SegmentedArraySize::kVariableLength) {
      auto byte_len = x.size() * sizeof(T*);
      allocation = paddle::memory::Alloc(
          ctx.GetPlace(),
          byte_len,
          phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
      dev_ptr = allocation->ptr();

      paddle::memory::Copy(ctx.GetPlace(),
                           dev_ptr,
                           phi::CPUPlace(),
                           reinterpret_cast<void*>(x_ptrs.data()),
                           x_ptrs.size() * sizeof(T*),
                           ctx.stream());
    }

    array.Set(x_ptrs, dev_ptr);
  }

 private:
  std::vector<const T*> x_ptrs;
  paddle::memory::AllocationPtr allocation{nullptr};
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
