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

void ScatterInferMeta(const MetaTensor& x,
                      const MetaTensor& index,
                      const MetaTensor& updates,
                      bool overwrite,
                      MetaTensor* out) {
  const auto& updates_dims = updates.dims();
  const auto& ref_dims = x.dims();
  const auto& index_dims = index.dims();
  PADDLE_ENFORCE_EQ(
      index_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The size of Input(Ids)'s shape should be equal to 1, but "
          "received the rank of Input(Ids) is %d.",
          index_dims.size()));
  PADDLE_ENFORCE_EQ(
      ref_dims.size(),
      updates_dims.size(),
      phi::errors::InvalidArgument(
          "Input(X) and Input(Updates) should have the same shape size, "
          "but received the size of Input(x)'s shape is %d, the size of "
          "Input(Updates)'s shape is %d.",
          ref_dims.size(),
          updates_dims.size()));
  PADDLE_ENFORCE_EQ(
      updates_dims[0],
      index_dims[0],
      phi::errors::InvalidArgument(
          "Input(Updates) and Input(Ids) should have same batch-size, but"
          " received Input(Updates)'s batch-size is %d, Input(Ids)'s "
          "batch-size is %d.",
          updates_dims[0],
          index_dims[0]));
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ScatterNdAddInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& updates,
                           MetaTensor* out) {
  const auto& ref_dims = x.dims();
  auto ref_dims_size = ref_dims.size();
  const auto& index_dims = index.dims();
  auto index_dims_size = index_dims.size();
  const auto& updates_dims = updates.dims();
  auto updates_dims_size = updates_dims.size();

  PADDLE_ENFORCE_LE(
      index_dims[index_dims_size - 1],
      ref_dims_size,
      phi::errors::InvalidArgument(
          "The last dimension of Input(Index)'s shape should be no greater "
          "than the rank of Input(X), but received the last dimension of "
          "Input(Index)'s shape is %d, the rank of Input(X) is %d.",
          index_dims[index_dims_size - 1],
          ref_dims_size));
  PADDLE_ENFORCE_GE(index_dims_size,
                    2UL,
                    phi::errors::InvalidArgument(
                        "The rank of Input(Index) should be greater than 1, "
                        "but received the rank of Input(Index) is %d.",
                        index_dims_size));

  // update.shape = index.shape[:-1] + output.shape[index.shape[-1]:]
  std::vector<int64_t> r_updates_dims;
  for (int64_t i = 0; i < index_dims_size - 1; ++i) {
    r_updates_dims.emplace_back(index_dims[i]);
  }
  for (int64_t i = index_dims[index_dims_size - 1]; i < ref_dims_size; ++i) {
    r_updates_dims.emplace_back(ref_dims[i]);
  }

  PADDLE_ENFORCE_EQ(
      r_updates_dims.size(),
      updates_dims_size,
      phi::errors::InvalidArgument(
          "Updates has wrong shape. The shape of Updates and Input(Updates) "
          "should be same, but received the shape of Updates is %d, "
          "the shape of Input(Updates) is %d.",
          r_updates_dims.size(),
          updates_dims_size));

  for (int64_t i = 0; i < updates_dims_size; ++i) {
    PADDLE_ENFORCE_EQ(
        r_updates_dims[i],
        updates_dims[i],
        phi::errors::InvalidArgument(
            "Updates has wrong shape. The dimensions of Updates and "
            "Input(Updates) should match, but received Updates's"
            "%d-th dimension is %d, Input(Updates)'s %d-th "
            "dimension is %d.",
            i,
            r_updates_dims[i],
            i,
            updates_dims[i]));
  }
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void LerpInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   const MetaTensor& weight,
                   MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto w_dims = weight.dims();
  DDim out_dims;
  out_dims = funcs::GetOutputDims(x_dims, y_dims);
  if (w_dims.size() > 1 || w_dims[0] != 1) {
    out_dims = funcs::GetOutputDims(out_dims, w_dims);
  }
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

}  // namespace phi
