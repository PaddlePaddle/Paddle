/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/backward.h"

namespace phi {

void BilinearTensorProductGradInferMeta(const MetaTensor& x,
                                        const MetaTensor& y,
                                        const MetaTensor& weight,
                                        const MetaTensor& dout,
                                        MetaTensor* dx,
                                        MetaTensor* dy,
                                        MetaTensor* dweight,
                                        MetaTensor* dbias) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto weight_dims = weight.dims();
  auto out_dims = dout.dims();

  PADDLE_ENFORCE_EQ(
      out_dims.size(),
      2UL,
      errors::InvalidArgument("The input(Out@GRAD) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      x_dims[0],
      out_dims[0],
      errors::InvalidArgument(
          "The first dimension(batch_size) of input(Out@GRAD) must be "
          "equal to the first dimension of the Input(X)."));
  PADDLE_ENFORCE_EQ(
      weight_dims[0],
      out_dims[1],
      errors::InvalidArgument(
          "The second dimension of input(Out@GRAD) must be equal to "
          "the third dimension of the Input(Weight)."));

  if (dx) {
    dx->set_dims(x_dims);
    dx->set_dtype(x.dtype());
  }
  if (dy) {
    dy->set_dims(y_dims);
    dy->set_dtype(y.dtype());
  }
  if (dweight) {
    dweight->set_dims(weight_dims);
    dweight->set_dtype(weight.dtype());
  }
  if (dbias) {
    dbias->set_dims({1, out_dims[1]});
    dbias->set_dtype(dout.dtype());
  }
}

void GeneralBinaryGradInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                MetaTensor* dx,
                                MetaTensor* dy) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy) {
    dy->share_meta(y);
  }
}

void GeneralTernaryGradInferMeta(const MetaTensor& x,
                                 const MetaTensor& y,
                                 const MetaTensor& z,
                                 MetaTensor* dx,
                                 MetaTensor* dy,
                                 MetaTensor* dz) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy) {
    dy->share_meta(y);
  }
  if (dz) {
    dz->share_meta(z);
  }
}

void GumbelSoftmaxGradInferMeta(const MetaTensor& out,
                                const MetaTensor& dout,
                                int axis,
                                MetaTensor* dx) {
  PADDLE_ENFORCE_EQ(
      out.dims(),
      dout.dims(),
      errors::InvalidArgument(
          "Input(Out) and its gradients should have the same shape."));
  dx->share_meta(dout);
}

}  // namespace phi
