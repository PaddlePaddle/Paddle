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

#include "paddle/phi/kernels/fill_diagonal_tensor_kernel.h"
#include <array>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

void CalMatDims(phi::DDim out_dims,
                int dim1,
                int dim2,
                int64_t *offset,
                int64_t *new_dims,
                int64_t *strides,
                int64_t *matoffset) {
  int64_t dimprod = 1, batchdim = 1;
  int rank = out_dims.size();
  int matoffidx = 0;
  for (int i = rank - 1; i >= 0; i--) {
    if (i == dim2) {
      strides[0] = dimprod;
    } else if (i == dim1) {
      strides[1] = dimprod;
    } else {
      batchdim *= out_dims[i];
      // matoffset calculate the offset position of the diagonal defined by dim1
      // and dim2
      // the first circle calculate the final free dimension
      // and then calculate the front free dim one by one
      if (matoffidx == 0) {
        for (int64_t j = 0; j < out_dims[i]; j++) {
          matoffset[matoffidx] = dimprod * j;
          matoffidx++;
        }
      } else {
        auto size = matoffidx;
        for (int64_t j = 1; j < out_dims[i]; j++) {
          for (int64_t k = 0; k < size; k++) {
            matoffset[matoffidx] = matoffset[k] + dimprod * j;
            matoffidx++;
          }
        }
      }
    }
    dimprod *= out_dims[i];
  }

  int64_t diagdim = dim1;
  if (*offset >= 0) {
    diagdim = std::min(out_dims[dim1], out_dims[dim2] - *offset);
    *offset *= strides[0];
  } else {
    diagdim = std::min(out_dims[dim1] + *offset, out_dims[dim2]);
    *offset *= -strides[1];
  }
  new_dims[0] = batchdim;
  new_dims[1] = diagdim;
  return;
}

template <typename T, typename Context>
void FillDiagonalTensorKernel(const Context &ctx,
                              const DenseTensor &x,
                              const DenseTensor &y,
                              int64_t offset,
                              int dim1,
                              int dim2,
                              DenseTensor *out) {
  T *out_data = ctx.template Alloc<T>(out);
  const T *fill_data = y.data<T>();

  phi::Copy(ctx, x, ctx.GetPlace(), false, out);
  auto out_dims = out->dims();
  const auto &matdims = y.dims();
  auto fill_dims = common::flatten_to_2d(matdims, matdims.size() - 1);

  std::array<int64_t, 2> new_dims = {};
  std::array<int64_t, 2> strides = {};
  std::vector<int64_t> matdim;
  matdim.resize(fill_dims[0]);
  CalMatDims(out_dims,
             dim1,
             dim2,
             &offset,
             new_dims.data(),
             strides.data(),
             matdim.data());
  PADDLE_ENFORCE_EQ(
      new_dims[0],
      fill_dims[0],
      errors::InvalidArgument("The dims should be %d x %d, but get "
                              "%d x %d in fill tensor Y",
                              new_dims[0],
                              new_dims[1],
                              fill_dims[0],
                              fill_dims[1]));
  PADDLE_ENFORCE_EQ(
      new_dims[1],
      fill_dims[1],
      errors::InvalidArgument("The dims should be %d x %d, but get "
                              "%d x %d in fill tensor Y",
                              new_dims[0],
                              new_dims[1],
                              fill_dims[0],
                              fill_dims[1]));

  auto size = out->numel();
  for (int64_t i = 0; i < fill_dims[0]; i += 1) {
    auto sumoff = matdim[i] + offset;
    for (int64_t j = 0; j < fill_dims[1]; j += 1) {
      auto fill_index = j * (strides[1] + strides[0]) + sumoff;
      if (fill_index < size) {
        out_data[fill_index] = fill_data[i * fill_dims[1] + j];
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalTensorKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   int16_t,
                   int8_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   bool) {}
