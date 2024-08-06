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

#include "paddle/phi/kernels/share_buffer_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {

template <typename Context>
void ShareBufferKernel(const Context &dev_ctx UNUSED,
                       const std::vector<const DenseTensor *> &x,
                       const std::vector<bool> &share_dims_and_dtype UNUSED,
                       std::vector<DenseTensor *> out,
                       std::vector<DenseTensor *> xout UNUSED) {
  PADDLE_ENFORCE_EQ(
      x.size(),
      out.size(),
      common::errors::PermissionDenied(
          "The input(X) and Output(out) should have the same size, but got "
          "size of Input(X) is %d and size of Output(out) is %d.",
          x.size(),
          out.size()));
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] == nullptr || out[i] == nullptr) {
      continue;
    }
    out[i]->ShareBufferWith(*x[i]);
    VLOG(10) << "Share tensor buffer ";
  }
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(share_buffer,
                                         ALL_LAYOUT,
                                         phi::ShareBufferKernel) {}
