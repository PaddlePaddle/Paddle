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

#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/for_range.h"

// TODO(paddle-dev): Remove this file when we can call related Kernel directly

namespace phi {
namespace funcs {

inline int ComputeStride(int axis, phi::DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T, typename ValueType>
struct DiagAndFillFunctor {
  DiagAndFillFunctor(const int m,
                     const int n,
                     const int num_lower_diags,
                     const int num_upper_diags,
                     const ValueType* scale,
                     const T* input,
                     T* output)
      : m_(m),
        n_(n),
        num_lower_diags_(num_lower_diags),
        num_upper_diags_(num_upper_diags),
        scale_(scale),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int col = index % n_;
    const int row = (index / n_) % m_;
    const int band_start = (num_lower_diags_ < 0 ? 0 : row - num_lower_diags_);
    const int band_end =
        (num_upper_diags_ < 0 ? n_ : row + num_upper_diags_ + 1);
    if (col < band_start || col >= band_end) {
      output_[index] = input_[index];
    } else if (col == band_end - 1) {
      output_[index] = static_cast<T>(scale_[index % m_]);
    } else {
      output_[index] = input_[index];
    }
  }

 private:
  const int m_, n_, num_lower_diags_, num_upper_diags_;
  const ValueType* scale_;
  const T* input_;
  T* output_;
};

template <typename T, typename ValueType, typename Context>
DenseTensor DiagFill(const Context& dev_ctx,
                     const int m,
                     const int n,
                     const int num_lower_diags,
                     const int num_upper_diags,
                     const DenseTensor& scale,
                     const DenseTensor& input) {
  DenseTensor out;
  out.Resize(input.dims());
  dev_ctx.template Alloc<T>(&out);
  funcs::ForRange<Context> for_range(dev_ctx, input.numel());
  DiagAndFillFunctor<T, ValueType> diag_and_copy_functor(
      m,
      n,
      num_lower_diags,
      num_upper_diags,
      scale.data<ValueType>(),
      input.data<T>(),
      out.data<T>());
  for_range(diag_and_copy_functor);
  return out;
}

template <typename T, typename Context>
DenseTensor BatchDiag(const Context& dev_ctx, const DenseTensor& x, int batch) {
  DenseTensor out;
  auto* x_data = x.data<phi::dtype::Real<T>>();
  auto numel = x.numel();
  out.Resize(x.dims());
  auto* out_data = dev_ctx.template HostAlloc<phi::dtype::Real<T>>(
      &out, static_cast<size_t>(numel * sizeof(phi::dtype::Real<T>)));

  auto x_dims = x.dims();
  int num_dims = x_dims.size();
  std::vector<int> out_shape;

  for (int i = 0; i < num_dims - 1; ++i) {
    out_shape.push_back(x.dims()[i]);
  }
  out.Resize(phi::make_ddim(out_shape));
  int order = x.dims()[num_dims - 1];
  int stride_out = order * order;
  int stride_in = order + 1;
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < order; ++j) {
      out_data[i * order + j] = x_data[stride_out * i + stride_in * j];
    }
  }
  return out;
}

}  // namespace funcs
}  // namespace phi
