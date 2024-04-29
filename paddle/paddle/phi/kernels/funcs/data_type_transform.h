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

#pragma once

#include "glog/logging.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace phi {
namespace funcs {

template <typename Context>
phi::DenseTensor TransDataType(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               DataType dtype) {
  VLOG(3) << "TransDataType "
          << "src type:" << x.dtype() << "; dst type: " << dtype;

  switch (x.dtype()) {
    case DataType::FLOAT32:
      return phi::Cast<float>(dev_ctx, x, dtype);
    case DataType::FLOAT64:
      return phi::Cast<double>(dev_ctx, x, dtype);
    case DataType::INT32:
      return phi::Cast<int32_t>(dev_ctx, x, dtype);
    case DataType::INT64:
      return phi::Cast<int64_t>(dev_ctx, x, dtype);
    case DataType::FLOAT16:
      return phi::Cast<phi::dtype::float16>(dev_ctx, x, dtype);
    case DataType::BFLOAT16:
      return phi::Cast<phi::dtype::bfloat16>(dev_ctx, x, dtype);
    case DataType::BOOL:
      return phi::Cast<bool>(dev_ctx, x, dtype);
    case DataType::INT16:
      return phi::Cast<int16_t>(dev_ctx, x, dtype);
    case DataType::UINT8:
      return phi::Cast<uint8_t>(dev_ctx, x, dtype);
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          x.dtype()));
  }
}

}  // namespace funcs
}  // namespace phi
