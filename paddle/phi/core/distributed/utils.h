// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace distributed {

inline phi::DenseTensor GetPartialTensor(const phi::DenseTensor& tensor,
                                         int64_t offset,
                                         int64_t numel) {
  phi::DenseTensor tensor_flattened;
  tensor_flattened.ShareDataWith(tensor);
  tensor_flattened.Resize({tensor.numel()});
  return tensor_flattened.Slice(offset, offset + numel);
}

#define NCCL_CHECK(cmd)     \
  do {                      \
    ncclResult_t r = cmd;   \
    if (r != ncclSuccess) { \
      exit(EXIT_FAILURE);   \
    }                       \
  } while (0)

}  //  namespace distributed
}  // namespace phi
