/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/transform.h"

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/functions/eigen/common.h"

namespace pten {
namespace math {

template <typename InT, typename OutT>
struct CastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <typename DeviceContext, typename InT>
struct CastOpFunctor {
  const DeviceContext& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  CastOpFunctor(const DenseTensor& in,
                DenseTensor* out,
                const DeviceContext& ctx)
      : ctx_(ctx), in_(in), out_(out) {}

  template <typename OutT>
  void apply() const {
    auto* in_begin = in_.data<InT>();
    auto numel = in_.numel();
    auto* in_end = in_begin + numel;
    auto* out_begin = out_->mutable_data<OutT>();
    paddle::platform::Transform<DeviceContext> trans;
    trans(
        ctx_, in_begin, in_end, out_begin, CastOpTransformFunctor<InT, OutT>());
  }
};

template <typename DeviceContext, typename T>
struct TransposeNormal {
  void operator()(const DeviceContext& context,
                  const DenseTensor& in,
                  DenseTensor* out,
                  const std::vector<int>& axis) {
    const int rank = axis.size();
    auto in_stride = paddle::framework::stride(in.dims());
    auto out_stride = paddle::framework::stride(out->dims());
    const T* in_ptr = in.data<T>();
    T* out_ptr = out->mutable_data<T>();

    auto transpose_helper = [&](int64_t beg, int64_t end) {
      for (int64_t out_idx = beg; out_idx < end; ++out_idx) {
        int64_t in_idx = 0;
        int64_t tmp_idx = out_idx;
        // calculate the input index
        for (int i = 0; i < rank; ++i) {
          const int64_t coordinate = tmp_idx / out_stride[i];
          tmp_idx -= coordinate * out_stride[i];
          in_idx += coordinate * in_stride[axis[i]];
        }
        out_ptr[out_idx] = in_ptr[in_idx];
      }
    };
    transpose_helper(0, out->numel());
  }
};

// define transpose normal
#define DEFINE_CPU_TRANS_NORMAL(TYPE) \
  template struct TransposeNormal<paddle::platform::CPUDeviceContext, TYPE>

DEFINE_CPU_TRANS_NORMAL(paddle::platform::float16);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::bfloat16);
DEFINE_CPU_TRANS_NORMAL(float);
DEFINE_CPU_TRANS_NORMAL(double);
DEFINE_CPU_TRANS_NORMAL(int);
DEFINE_CPU_TRANS_NORMAL(int64_t);
DEFINE_CPU_TRANS_NORMAL(bool);
DEFINE_CPU_TRANS_NORMAL(int16_t);
DEFINE_CPU_TRANS_NORMAL(uint8_t);
DEFINE_CPU_TRANS_NORMAL(int8_t);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::complex<float>);
DEFINE_CPU_TRANS_NORMAL(paddle::platform::complex<double>);

}  // namespace math
}  // namespace pten
