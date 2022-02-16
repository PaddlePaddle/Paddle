// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/kernels/funcs/algorithm.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

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
  static HOSTDEVICE bool IsNan(int x) { return false; }
  static HOSTDEVICE bool IsNan(int64_t x) { return false; }

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
  static HOSTDEVICE bool IsInf(int x) { return false; }
  static HOSTDEVICE bool IsInf(int64_t x) { return false; }

  HOSTDEVICE GpuAndCpuSearchSortedCompute(const T1* sequence_data,
                                          const T2* value_data, bool right,
                                          bool is_1d_boundaries,
                                          int64_t val_size, int64_t seq_size,
                                          OutType* out_data)
      : sequence_data_(sequence_data),
        value_data_(value_data),
        right_(right),
        is_1d_boundaries_(is_1d_boundaries),
        val_size_(val_size),
        seq_size_(seq_size),
        out_data_(out_data) {}
  HOSTDEVICE void operator()(int64_t idx) {
    const T2* value_ptr = value_data_ + idx;
    const T1* sequence_ptr = is_1d_boundaries_
                                 ? sequence_data_
                                 : sequence_data_ + idx / val_size_ * seq_size_;
    if (IsInf(*value_ptr) || IsNan(*value_ptr)) {
      out_data_[idx] = seq_size_;
    } else {
      if (right_) {
        out_data_[idx] = static_cast<OutType>(pten::funcs::UpperBound<T1, T2>(
            sequence_ptr, seq_size_, *value_ptr));
      } else {
        out_data_[idx] = static_cast<OutType>(pten::funcs::LowerBound<T1, T2>(
            sequence_ptr, seq_size_, *value_ptr));
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

template <typename DeviceContext, typename T1, typename OutType>
class SearchSortedFunctor {
 public:
  SearchSortedFunctor(const framework::ExecutionContext& context,
                      const framework::Tensor* sorted_sequence,
                      const framework::Tensor* value, bool right,
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
    const framework::DDim& seq_dims = sorted_sequence_->dims();
    const framework::DDim& val_dims = value_->dims();

    bool is_1d_boundaries = seq_dims.size() == 1;
    int64_t val_size = val_dims[val_dims.size() - 1];
    int64_t seq_size = seq_dims[seq_dims.size() - 1];

    auto& dev_ctx = context_.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, value_->numel());
    GpuAndCpuSearchSortedCompute<T1, T2, OutType>
        gpu_and_cpu_search_sorted_compute(sequence_data, value_data, right_,
                                          is_1d_boundaries, val_size, seq_size,
                                          out_data_);
    for_range(gpu_and_cpu_search_sorted_compute);
  }

 private:
  const framework::ExecutionContext& context_;
  const framework::Tensor* sorted_sequence_;
  const framework::Tensor* value_;
  bool right_;
  OutType* out_data_;
};

template <typename Visitor>
static void VisitDataType(framework::proto::VarType::Type type,
                          Visitor visitor) {
  if (type == framework::proto::VarType::FP32) {
    visitor.template apply<float>();
  } else if (type == framework::proto::VarType::FP64) {
    visitor.template apply<double>();
  } else if (type == framework::proto::VarType::INT32) {
    visitor.template apply<int>();
  } else if (type == framework::proto::VarType::INT64) {
    visitor.template apply<int64_t>();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The recieved values data type %s can not meet input requirements. "
        "Because the given values data type of searchsorted operators must be "
        "float32, float64, int32 or int64. Please input appropriate "
        "sorted_sequence again! ",
        framework::DataTypeToString(type)));
  }
}

template <typename DeviceContext, typename T>
class SearchSortedKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* sorted_sequence = context.Input<Tensor>("SortedSequence");
    auto* value = context.Input<Tensor>("Values");
    bool out_int32 = context.Attr<bool>("out_int32");
    bool right = context.Attr<bool>("right");
    auto* out = context.Output<Tensor>("Out");

    if (out_int32) {
      int* out_data = out->mutable_data<int>(context.GetPlace());
      SearchSortedFunctor<DeviceContext, T, int> functor(
          context, sorted_sequence, value, right, out_data);
      VisitDataType(framework::TransToProtoVarType(value->dtype()), functor);
    } else {
      int64_t* out_data = out->mutable_data<int64_t>(context.GetPlace());
      SearchSortedFunctor<DeviceContext, T, int64_t> functor(
          context, sorted_sequence, value, right, out_data);
      VisitDataType(framework::TransToProtoVarType(value->dtype()), functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle
