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

#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

void AddmmInferMeta(const MetaTensor& input,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    float alpha,
                    float beta,
                    MetaTensor* out) {
  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  auto ndim_input = input_dims.size();
  auto ndim_x = x_dims.size();
  auto ndim_y = y_dims.size();

  VLOG(3) << "addmm operator input.shape=" << input_dims
          << " x.shape=" << x_dims << " y.shape=" << y_dims << " beta=" << beta
          << " alpha=" << alpha << " ndim_input=" << ndim_input
          << " ndim_x=" << ndim_x << " ndim_y=" << ndim_y;

  PADDLE_ENFORCE_NE(
      product(input_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'input' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(x_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'x' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(y_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'y' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));
  // dim check
  PADDLE_ENFORCE_EQ(
      ndim_input,
      2,
      errors::InvalidArgument("The input tensor input's dimension must be 2. "
                              "But received input's dimension = [%s].",
                              ndim_input));
  PADDLE_ENFORCE_EQ(
      ndim_x,
      2,
      errors::InvalidArgument("The input tensor x's dimension must be 2. "
                              "But received x's dimension = [%s].",
                              ndim_x));
  PADDLE_ENFORCE_EQ(
      ndim_y,
      2,
      errors::InvalidArgument("The input tensor y's dimension must be 2. "
                              "But received y's dimension = [%s].",
                              ndim_y));

  std::vector<int64_t> output_dims;
  output_dims.push_back(x_dims[0]);
  output_dims.push_back(y_dims[1]);

  out->set_dims(make_ddim(output_dims));
  out->share_lod(input);
  out->set_dtype(input.dtype());
}

}  // namespace phi
