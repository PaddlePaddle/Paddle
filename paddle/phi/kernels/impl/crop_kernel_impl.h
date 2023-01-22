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

#include "paddle/phi/kernels/crop_kernel.h"

#include <utility>
#include <vector>

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

static phi::DDim ValidateShape(const std::vector<int64_t>& shape,
                               const std::vector<int64_t>& offsets,
                               const phi::DDim& in_dims) {
  auto in_dim_size = in_dims.size();
  auto shape_size = shape.size();
  PADDLE_ENFORCE_EQ(
      in_dim_size,
      shape_size,
      errors::InvalidArgument(
          "The number of elements (%d) for shape of Op(crop_tensor) should be "
          "equal to the number of dimensions (%d) of the input tensor.",
          shape_size,
          in_dim_size));
  std::vector<int64_t> output_shape(shape.size(), 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 && in_dims[i] > 0) {
      PADDLE_ENFORCE_NE(shape[i],
                        0,
                        errors::InvalidArgument(
                            "The value (%d) of the %uth element for shape of "
                            "Op(crop_tensor) should not be zero.",
                            shape[i],
                            i));
      PADDLE_ENFORCE_EQ(
          shape[i],
          -1,
          errors::InvalidArgument("When the value (%d) of the %uth "
                                  "element for shape of Op(crop_tensor)"
                                  " is negative, only -1 is supported.",
                                  shape[i],
                                  i));
      output_shape[i] = in_dims[i] - offsets[i];
    } else {
      output_shape[i] = static_cast<int64_t>(shape[i]);
    }
  }

  return phi::make_ddim(output_shape);
}

template <typename Context, typename T, size_t D>
void CropTensorFunction(const Context& dev_ctx,
                        const DenseTensor& x,
                        const IntArray& shape,
                        const IntArray& offsets,
                        DenseTensor* out) {
  auto x_dims = x.dims();
  auto rank = x.dims().size();
  auto out_dims = out->dims();

  auto shape_vec = shape.GetData();

  if (shape_vec.size() == 0) {
    for (int i = 0; i < out_dims.size(); ++i) {
      shape_vec.push_back(out_dims[i]);
    }
  }

  auto offsets_vec = offsets.GetData();

  PADDLE_ENFORCE_EQ(
      rank,
      static_cast<int>(offsets_vec.size()),
      errors::InvalidArgument("The number of elements (%d) for "
                              "input 'Offsets' must be equal to "
                              "the number of dimensions (%d) "
                              "of the input tensor.",
                              static_cast<int>(offsets_vec.size()),
                              rank));

  out_dims = ValidateShape(shape_vec, offsets_vec, x.dims());
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  for (size_t i = 0; i < offsets_vec.size(); ++i) {
    PADDLE_ENFORCE_GE(
        offsets_vec[i],
        0,
        errors::InvalidArgument("The offsets (%d) of the %uth elements of"
                                " Op(crop_tensor) "
                                "should be greater than or "
                                "equal to 0.",
                                offsets_vec[i],
                                i));

    PADDLE_ENFORCE_LE(offsets_vec[i] + shape_vec[i],
                      x_dims[i],
                      errors::InvalidArgument(
                          "The sum of the %uth elements of "
                          "offsets (%d) and shape (%d) of Op(crop_tensor) "
                          "should be less than or "
                          "equal to the size of %uth dimension of the input.",
                          i,
                          offsets_vec[i],
                          shape_vec[i],
                          i));
  }

  auto x_tensor = EigenTensor<T, D>::From(x);
  auto out_tensor = EigenTensor<T, D>::From(*out);
  Eigen::DSizes<Eigen::DenseIndex, D> e_offsets;
  Eigen::DSizes<Eigen::DenseIndex, D> e_shape;
  for (size_t i = 0; i < D; ++i) {
    e_offsets[i] = offsets_vec[i];
    e_shape[i] = out->dims()[i];
  }
  auto& place = *dev_ctx.eigen_device();
  phi::funcs::EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(
      place, out_tensor, x_tensor, e_offsets, e_shape);
}

template <typename T, typename Context>
void CropKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& shape,
                const IntArray& offsets,
                DenseTensor* out) {
  int rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1,
      errors::InvalidArgument(
          "The number of dimensions of the input 'x' for "
          "Op(crop_tensor) must be greater than or equal to 1, but the "
          "value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      errors::InvalidArgument(
          "The number of dimensions of the input 'x' for "
          "Op(crop_tensor) must be less than or equal to 6, but the "
          "value received is %d.",
          rank));
  switch (rank) {
    case 1:
      CropTensorFunction<Context, T, 1>(dev_ctx, x, shape, offsets, out);
      break;
    case 2:
      CropTensorFunction<Context, T, 2>(dev_ctx, x, shape, offsets, out);
      break;
    case 3:
      CropTensorFunction<Context, T, 3>(dev_ctx, x, shape, offsets, out);
      break;
    case 4:
      CropTensorFunction<Context, T, 4>(dev_ctx, x, shape, offsets, out);
      break;
    case 5:
      CropTensorFunction<Context, T, 5>(dev_ctx, x, shape, offsets, out);
      break;
    case 6:
      CropTensorFunction<Context, T, 6>(dev_ctx, x, shape, offsets, out);
      break;
  }
}

}  // namespace phi
