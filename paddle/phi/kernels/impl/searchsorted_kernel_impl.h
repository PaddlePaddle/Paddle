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

#include <math.h>

#include "paddle/common/ddim.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T1, typename T2, typename OutType>
class GpuAndCpuSearchSortedCompute {
 public:
  static HOSTDEVICE bool IsNan(float x) {
#ifdef __NVCC__
    return ::isnan(x);
#else
    return std::isnan(x);
#endif
  }
  static HOSTDEVICE bool IsNan(double x) {
#ifdef __NVCC__
    return ::isnan(x);
#else
    return std::isnan(x);
#endif
  }
  static HOSTDEVICE bool IsNan(int x UNUSED) { return false; }
  static HOSTDEVICE bool IsNan(int64_t x UNUSED) { return false; }

  static HOSTDEVICE bool IsInf(float x) {
#ifdef __NVCC__
    return ::isinf(x);
#else
    return std::isinf(x);
#endif
  }
  static HOSTDEVICE bool IsInf(double x) {
#ifdef __NVCC__
    return ::isinf(x);
#else
    return std::isinf(x);
#endif
  }
  static HOSTDEVICE bool IsInf(int x UNUSED) { return false; }
  static HOSTDEVICE bool IsInf(int64_t x UNUSED) { return false; }

  HOSTDEVICE inline size_t LowerBound(const T1* x, size_t num, const T2& val) {
    // @{ Group LowerBound
    // The following code is from
    // https://en.cppreference.com/w/cpp/algorithm/lower_bound
    using MT1 = typename phi::dtype::MPTypeTrait<T1>::Type;
    using MT2 = typename phi::dtype::MPTypeTrait<T2>::Type;
    MT2 val_mt = static_cast<MT2>(val);

    auto* first = x;
    int64_t count = static_cast<int64_t>(num);
    while (count > 0) {
      int64_t step = (count >> 1);
      auto* it = first + step;
      MT1 it_mt = static_cast<MT1>(*it);
      if (it_mt < val_mt) {
        first = ++it;
        count -= (step + 1);
      } else {
        count = step;
      }
    }
    return static_cast<size_t>(first - x);
  }

  HOSTDEVICE inline size_t UpperBound(const T1* x, size_t num, const T2& val) {
    // @{ Group UpperBound
    // The following code is from
    // https://en.cppreference.com/w/cpp/algorithm/upper_bound
    using MT1 = typename phi::dtype::MPTypeTrait<T1>::Type;
    using MT2 = typename phi::dtype::MPTypeTrait<T2>::Type;
    MT2 val_mt = static_cast<MT2>(val);

    auto* first = x;
    int64_t count = static_cast<int64_t>(num);
    while (count > 0) {
      auto step = (count >> 1);
      auto* it = first + step;
      MT1 it_mt = static_cast<MT1>(*it);
      if (val_mt < it_mt) {
        count = step;
      } else {
        first = ++it;
        count -= (step + 1);
      }
    }
    return static_cast<size_t>(first - x);
  }

  HOSTDEVICE GpuAndCpuSearchSortedCompute(const T1* sequence_data,
                                          const T2* value_data,
                                          bool right,
                                          bool is_1d_boundaries,
                                          int64_t val_size,
                                          int64_t seq_size,
                                          OutType* out_data)
      : sequence_data_(sequence_data),
        value_data_(value_data),
        right_(right),
        is_1d_boundaries_(is_1d_boundaries),
        val_size_(val_size),
        seq_size_(seq_size),
        out_data_(out_data) {}
  HOSTDEVICE void operator()(int64_t idx) {
    using MT2 = typename phi::dtype::MPTypeTrait<T2>::Type;
    const T2* value_ptr = value_data_ + idx;
    const MT2 value_mt = static_cast<MT2>(*value_ptr);
    const T1* sequence_ptr = is_1d_boundaries_
                                 ? sequence_data_
                                 : sequence_data_ + idx / val_size_ * seq_size_;
    if (IsInf(value_mt) || IsNan(value_mt)) {
      out_data_[idx] = seq_size_;
    } else {
      if (right_) {
        out_data_[idx] = static_cast<OutType>(
            UpperBound(sequence_ptr, seq_size_, *value_ptr));
      } else {
        out_data_[idx] = static_cast<OutType>(
            LowerBound(sequence_ptr, seq_size_, *value_ptr));
      }
    }
  }

 private:
  const T1* sequence_data_;
  const T2* value_data_;
  bool right_;
  bool is_1d_boundaries_;
  int64_t val_size_;
  int64_t seq_size_;
  OutType* out_data_;
};

template <typename Context, typename T1, typename OutType>
class SearchSortedFunctor {
 public:
  SearchSortedFunctor(const Context& context,
                      const DenseTensor* sorted_sequence,
                      const DenseTensor* value,
                      bool right,
                      OutType* out_data)
      : context_(context),
        sorted_sequence_(sorted_sequence),
        value_(value),
        right_(right),
        out_data_(out_data) {}

  template <typename T2>
  void apply() {
    const T1* sequence_data = sorted_sequence_->data<T1>();
    const T2* value_data = value_->data<T2>();
    const phi::DDim& seq_dims = sorted_sequence_->dims();
    const phi::DDim& val_dims = value_->dims();

    bool is_1d_boundaries = seq_dims.size() == 1;
    int64_t val_size = 0;
    int64_t seq_size = 0;
    if (val_dims.size()) {
      val_size = val_dims[val_dims.size() - 1];
    } else {
      val_size = 1;
    }
    if (seq_dims.size()) {
      seq_size = seq_dims[seq_dims.size() - 1];
    } else {
      seq_size = 1;
    }

    funcs::ForRange<Context> for_range(context_, value_->numel());
    GpuAndCpuSearchSortedCompute<T1, T2, OutType>
        gpu_and_cpu_search_sorted_compute(sequence_data,
                                          value_data,
                                          right_,
                                          is_1d_boundaries,
                                          val_size,
                                          seq_size,
                                          out_data_);
    for_range(gpu_and_cpu_search_sorted_compute);
  }

 private:
  const Context& context_;
  const DenseTensor* sorted_sequence_;
  const DenseTensor* value_;
  bool right_;
  OutType* out_data_;
};

template <typename Visitor>
void VisitDataTypeForSearchSorted(DataType type, Visitor visitor) {
  if (type == DataType::FLOAT32) {
    visitor.template apply<float>();
  } else if (type == DataType::FLOAT64) {
    visitor.template apply<double>();
  } else if (type == DataType::INT32) {
    visitor.template apply<int>();
  } else if (type == DataType::INT64) {
    visitor.template apply<int64_t>();
  } else if (type == DataType::FLOAT16) {
    visitor.template apply<phi::dtype::float16>();
  } else if (type == DataType::BFLOAT16) {
    visitor.template apply<phi::dtype::bfloat16>();
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "The received values data type %s can not meet input requirements. "
        "Because the given values data type of searchsorted operators must be "
        "bfloat16, float16, float32, float64, int32 or int64. Please input "
        "appropriate "
        "sorted_sequence again! ",
        type));
  }
}

template <typename T, typename Context>
void SearchsortedKernel(const Context& ctx,
                        const DenseTensor& sorted_sequence,
                        const DenseTensor& value,
                        bool out_int32,
                        bool right,
                        DenseTensor* out) {
  if (out_int32) {
    ctx.template Alloc<int>(out);
    int* out_data = out->data<int>();
    SearchSortedFunctor<Context, T, int> functor(
        ctx, &sorted_sequence, &value, right, out_data);
    VisitDataTypeForSearchSorted(value.dtype(), functor);
  } else {
    ctx.template Alloc<int64_t>(out);
    int64_t* out_data = out->data<int64_t>();
    SearchSortedFunctor<Context, T, int64_t> functor(
        ctx, &sorted_sequence, &value, right, out_data);
    VisitDataTypeForSearchSorted(value.dtype(), functor);
  }
}

}  // namespace phi
