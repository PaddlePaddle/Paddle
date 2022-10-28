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

#include "paddle/phi/infermeta/sparse/binary.h"

namespace phi {
namespace sparse {

inline void GetOutShape(const DDim& x_dims,
                        const std::vector<int>& kernel_sizes,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        DDim* out_dims) {
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      5,
      phi::errors::InvalidArgument("the shape of x should be (N, D, H, W, C)"));
  PADDLE_ENFORCE_EQ(kernel_sizes.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "the shape of kernel should be (D, H, W, C, OC)"));

  // infer out shape
  (*out_dims)[0] = x_dims[0];
  (*out_dims)[4] = kernel_sizes[4];
  for (int i = 1; i < 4; i++) {
    (*out_dims)[i] = (x_dims[i] + 2 * paddings[i - 1] -
                      dilations[i - 1] * (kernel_sizes[i - 1] - 1) - 1) /
                         strides[i - 1] +
                     1;
  }
}

inline void ResetSubmKernelSizeAndStrides(const DDim& kernel_dims,
                                          std::vector<int>* paddings,
                                          std::vector<int>* strides) {
  for (uint64_t i = 0; i < paddings->size(); i++) {
    (*paddings)[i] = kernel_dims[i] / 2;
    (*strides)[i] = 1;
  }
}

void Conv3dInferMeta(const MetaTensor& x,
                     const MetaTensor& kernel,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const int groups,
                     const bool subm,
                     const std::string& key,
                     MetaTensor* out,
                     MetaTensor* rulebook,
                     MetaTensor* counter) {
  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  DDim out_dims = {1, 1, 1, 1, 1};

  std::vector<int> kernel_sizes(kernel_dims.size());
  for (int i = 0; i < kernel_dims.size(); i++) {
    kernel_sizes[i] = kernel_dims[i];
  }

  std::vector<int> subm_paddings(paddings), subm_strides(strides);
  if (subm) {
    // the out shape of subm_conv is same as input shape
    // reset the padding=kernel_size/2 and strides=1
    ResetSubmKernelSizeAndStrides(kernel.dims(), &subm_paddings, &subm_strides);
  }

  GetOutShape(
      x_dims, kernel_sizes, subm_paddings, dilations, subm_strides, &out_dims);

  out->set_dtype(x.dtype());
  out->set_dims(out_dims);
  out->set_layout(x.layout());

  rulebook->set_dtype(DataType::INT32);
  rulebook->set_layout(DataLayout::NCHW);
  rulebook->set_dims({1});

  counter->set_dtype(DataType::INT32);
  counter->set_layout(DataLayout::NCHW);
  counter->set_dims({1});
}

inline const std::vector<int> PoolResetKernel(
    const std::vector<int>& kernel_sizes,
    const int in_channels,
    const int out_channels) {
  std::vector<int> res(kernel_sizes);
  res.resize(5);
  res[3] = in_channels;
  res[4] = out_channels;
  return res;
}

void Pool3dInferMeta(const MetaTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     MetaTensor* out,
                     MetaTensor* rulebook,
                     MetaTensor* counter) {
  const auto& x_dims = x.dims();
  DDim out_dims = {1, 1, 1, 1, 1};

  const std::vector<int>& real_kernel_sizes =
      PoolResetKernel(kernel_sizes, x_dims[4], x_dims[4]);
  GetOutShape(
      x_dims, real_kernel_sizes, paddings, dilations, strides, &out_dims);
  out->set_dtype(x.dtype());
  out->set_dims(out_dims);
  out->set_layout(x.layout());

  rulebook->set_dtype(DataType::INT32);
  rulebook->set_layout(DataLayout::NCHW);
  rulebook->set_dims({1});

  counter->set_dtype(DataType::INT32);
  counter->set_layout(DataLayout::NCHW);
  counter->set_dims({1});
}

void SparseCooTensorInferMeta(const MetaTensor& values,
                              const MetaTensor& indices,
                              const IntArray& dense_shape,
                              MetaTensor* out) {
  out->set_dims(phi::make_ddim(dense_shape.GetData()));
  out->set_dtype(values.dtype());
  out->set_layout(values.layout());
}

}  // namespace sparse
}  // namespace phi
