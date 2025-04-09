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

#include <stdint.h>

#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "paddle/phi/core/framework/dense_tensor_serialize.h"
#include "paddle/phi/core/framework/var_type_helper.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace phi {

template <typename T, typename Context>
void SaveKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const std::string& file_path,
                bool overwrite,
                bool save_as_fp16) {
  PADDLE_ENFORCE_EQ(
      FileExists(file_path) && !overwrite,
      false,
      common::errors::PreconditionNotMet(
          "%s exists!, cannot save to it when overwrite is set to false.",
          file_path,
          overwrite));

  MkDirRecursively(DirName(file_path).c_str());

  // FIXME(yuyang18): We save variable to local file now, but we should change
  // it to save an output stream.
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    common::errors::Unavailable(
                        "Cannot open %s to save variables.", file_path));

  auto in_dtype = x.dtype();
  auto out_dtype = save_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;

  if (in_dtype != out_dtype) {
    auto out = phi::Cast<T>(dev_ctx, x, out_dtype);
    phi::SerializeToStream(fout, out, dev_ctx);
  } else {
    phi::SerializeToStream(fout, x, dev_ctx);
  }
  fout.close();
}
}  // namespace phi
