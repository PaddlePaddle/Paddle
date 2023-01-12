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

#include "paddle/phi/kernels/funcs/fast_divmod.h"

namespace phi {
namespace funcs {

template <typename IndexT>
struct GeneralDivMod {
 public:
  explicit GeneralDivMod(IndexT d) { divmoder = phi::funcs::FastDivMod(d); }
  __device__ inline phi::funcs::FastDivMod::DivModT div_mod(IndexT val) {
    return divmoder.Divmod(val);
  }

  phi::funcs::FastDivMod divmoder;
};

template <>
struct GeneralDivMod<int64_t> {
 public:
  using DivModT = phi::AlignedVector<int64_t, 2>;

  explicit GeneralDivMod(int64_t d) { divisor = d; }
  __device__ inline DivModT div_mod(int64_t val) {
    DivModT data;
    data[0] = val / divisor;
    data[1] = val - data[0] * divisor;
    return data;
  }

  int64_t divisor;
};

#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x)
#endif

enum class SegmentedArraySize {
  kVariableLength = 0,
  kFixed4 = 4,
  kFixed8 = 8,
  kFixed16 = 16,
  kFixed32 = 32,
  kFixed64 = 64,
};

template <typename T, SegmentedArraySize Size>
struct PADDLE_ALIGN(256) ConstPointerArray {
 public:
  const T* data[static_cast<int>(Size)];

  void Set(const std::vector<const T*>& ptrs, const T** dev_ptr = nullptr) {
    for (auto i = 0; i < ptrs.size(); ++i) {
      data[i] = ptrs[i];
    }
  }
};

template <typename T>
struct PADDLE_ALIGN(256)
    ConstPointerArray<T, SegmentedArraySize::kVariableLength> {
 public:
  const T** data{nullptr};

  void Set(const std::vector<const T*>& ptrs, const T** dev_ptr = nullptr) {
    data = dev_ptr;
  }
};

template <typename T, SegmentedArraySize Size>
struct PADDLE_ALIGN(256) PointerArray {
 public:
  T* data[static_cast<int>(Size)];

  void Set(const std::vector<T*>& ptrs, T** dev_ptr = nullptr) {
    for (auto i = 0; i < ptrs.size(); ++i) {
      data[i] = ptrs[i];
    }
  }
};

template <typename T>
struct PADDLE_ALIGN(256) PointerArray<T, SegmentedArraySize::kVariableLength> {
 public:
  T** data{nullptr};

  void Set(const std::vector<T*>& ptrs, T** dev_ptr = nullptr) {
    data = dev_ptr;
  }
};

#undef PADDLE_ALIGN

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

template <typename Context, typename T, SegmentedArraySize Size>
struct ConstPointerArraySetter : public ArraySetterBase<Context> {
 public:
  ConstPointerArray<T, Size> array;

  ConstPointerArraySetter(const Context& ctx,
                          const std::vector<const DenseTensor*>& t) {
    ptrs.resize(t.size());
    for (int i = 0; i < t.size(); ++i) {
      ptrs[i] = t[i]->data<T>();
    }

    const T** dev_ptr = nullptr;
    if (Size == SegmentedArraySize::kVariableLength) {
      size_t num_bytes = t.size() * sizeof(T*);
      dev_ptr =
          reinterpret_cast<const T**>(ArraySetterBase<Context>::AllocAndCopy(
              ctx, reinterpret_cast<void*>(ptrs.data()), num_bytes));
    }

    array.Set(ptrs, dev_ptr);
  }

 private:
  std::vector<const T*> ptrs;
};

template <typename Context, typename T, SegmentedArraySize Size>
struct PointerArraySetter : public ArraySetterBase<Context> {
 public:
  PointerArray<T, Size> array;

  PointerArraySetter(const Context& ctx, std::vector<DenseTensor*>* t) {
    ptrs.resize(t->size());
    for (int i = 0; i < t->size(); ++i) {
      if (t->at(i) && (t->at(i)->numel() > 0)) {
        ptrs[i] = ctx.template Alloc<T>(t->at(i));
      } else {
        ptrs[i] = nullptr;
      }
    }

    T** dev_ptr = nullptr;
    if (Size == SegmentedArraySize::kVariableLength) {
      size_t num_bytes = t->size() * sizeof(T*);
      dev_ptr = reinterpret_cast<T**>(ArraySetterBase<Context>::AllocAndCopy(
          ctx, reinterpret_cast<void*>(ptrs.data()), num_bytes));
    }

    array.Set(ptrs, dev_ptr);
  }

 private:
  std::vector<T*> ptrs;
};

inline SegmentedArraySize CalcArraySize(int n) {
  if (n <= 4) {
    return SegmentedArraySize::kFixed4;
  } else if (n <= 8) {
    return SegmentedArraySize::kFixed8;
  } else if (n <= 16) {
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

#define _SEGMENTED_ARRAY_KERNEL_CASE(size, ...) \
  case (size): {                                \
    constexpr auto kArraySize = (size);         \
    __VA_ARGS__;                                \
  } break

#define _SEGMENTED_ARRAY_KERNEL_DEFAULT(size, ...) \
  default: {                                       \
    constexpr auto kArraySize = (size);            \
    __VA_ARGS__;                                   \
  } break

#define SEGMENTED_ARRAY_KERNEL_HELPER(...)                                    \
  _SEGMENTED_ARRAY_KERNEL_CASE(funcs::SegmentedArraySize::kFixed4,            \
                               ##__VA_ARGS__);                                \
  _SEGMENTED_ARRAY_KERNEL_CASE(funcs::SegmentedArraySize::kFixed8,            \
                               ##__VA_ARGS__);                                \
  _SEGMENTED_ARRAY_KERNEL_CASE(funcs::SegmentedArraySize::kFixed16,           \
                               ##__VA_ARGS__);                                \
  _SEGMENTED_ARRAY_KERNEL_CASE(funcs::SegmentedArraySize::kFixed32,           \
                               ##__VA_ARGS__);                                \
  _SEGMENTED_ARRAY_KERNEL_CASE(funcs::SegmentedArraySize::kFixed64,           \
                               ##__VA_ARGS__);                                \
  _SEGMENTED_ARRAY_KERNEL_DEFAULT(funcs::SegmentedArraySize::kVariableLength, \
                                  ##__VA_ARGS__);

}  // namespace phi
