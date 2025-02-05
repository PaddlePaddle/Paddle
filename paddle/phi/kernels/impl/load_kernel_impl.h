// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/phi/core/framework/dense_tensor_serialize.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace phi {

template <typename T, typename Context>
void LoadKernel(const Context& dev_ctx,
                const std::string& file_path,
                int64_t seek,
                const std::vector<int64_t>& shape,
                bool load_as_fp16,
                phi::DenseTensor* out) {
  // FIXME(yuyang18): We save variable to local file now, but we should change
  // it to save an output stream.
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fin),
      true,
      errors::Unavailable("Load operator fail to open file %s, please check "
                          "whether the model file is complete or damaged.",
                          file_path));
  PADDLE_ENFORCE_NOT_NULL(
      out,
      errors::InvalidArgument("The variable to be loaded cannot be found."));

  if (seek != -1) {
    PADDLE_ENFORCE_GE(seek,
                      0,
                      errors::InvalidArgument(
                          "seek with tensor must great than or equal to 0"));
    phi::DeserializeFromStream(fin, out, dev_ctx, seek, shape);
  } else {
    phi::DeserializeFromStream(fin, out, dev_ctx);
  }

  auto in_dtype = out->dtype();
  auto out_dtype = load_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;
  if (in_dtype != out_dtype) {
    phi::CastKernel<T>(dev_ctx, *out, out_dtype, out);
  }
}

}  // namespace phi
