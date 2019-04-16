/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

static inline framework::DDim ValidateShape(const std::vector<int> shape,
                                            const framework::DDim &in_dims) {
  const int64_t in_size = framework::product(in_dims);
  auto in_dims_vec = framework::vectorize(in_dims);
  bool all_positive = std::all_of(in_dims_vec.cbegin(), in_dims_vec.cend(),
                                  [](int64_t i) { return i > 0; });
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PADDLE_ENFORCE(unk_dim_idx == -1,
                     "Only one input dimension of Attr(shape) can be unknown.");
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE(
          static_cast<int>(i) < in_dims.size(),
          "The index of dimension to copy from input shape must be less "
          "than the size of input shape.");
    } else {
      PADDLE_ENFORCE(
          shape[i] > 0,
          "Each input dimension of Attr(shape) must not be negtive except "
          "one unknown dimension.");
    }

    capacity *= (shape[i] ? shape[i] : in_dims[i]);
    output_shape[i] = (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // in_size < 0 and is un-determinate in compile time, skip the check,
      // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, in_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -in_size / capacity;
      PADDLE_ENFORCE_EQ(output_shape[unk_dim_idx] * capacity, -in_size,
                        "Invalid shape is given.");
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    PADDLE_ENFORCE_EQ(capacity, in_size, "Invalid shape is given.");
  }
  return framework::make_ddim(output_shape);
}

static inline void ReshapeFunc(const framework::ExecutionContext &ctx) {
  auto *out = ctx.Output<framework::LoDTensor>("Out");
  auto *in = ctx.Input<framework::LoDTensor>("X");

  auto *shape_tensor = ctx.HasInput("Shape")
                           ? ctx.Input<framework::LoDTensor>("Shape")
                           : nullptr;

  framework::DDim out_dims = out->dims();

  if (shape_tensor) {
    auto *shape_data = shape_tensor->data<int>();
    framework::Tensor cpu_shape_tensor;
    if (platform::is_gpu_place(shape_tensor->place())) {
      TensorCopySync(*shape_tensor, platform::CPUPlace(), &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
    auto shape =
        std::vector<int>(shape_data, shape_data + shape_tensor->numel());
    out_dims = ValidateShape(shape, in->dims());
  }

  out->mutable_data(ctx.GetPlace(), in->type());
  framework::TensorCopy(*in, ctx.GetPlace(),
                        ctx.template device_context<platform::DeviceContext>(),
                        out);
  out->Resize(out_dims);
}
}  // namespace operators
}  // namespace paddle
