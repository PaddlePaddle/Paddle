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

#include "paddle/phi/kernels/reshape_kernel.h"
#include <algorithm>
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/contiguous_kernel.h"

namespace phi {

bool ReshapeStride(const DDim& old_dims,
                   const DDim& old_stride,
                   const DDim& new_dims,
                   DDim& new_stride) {  // NOLINT
  int64_t numel = product(old_dims);
  if (numel < 0) {
    int64_t tmp[2];
    tmp[0] = 1;
    tmp[1] = new_dims.size();
    new_stride = DDim(tmp, 2);
    return true;
  } else if (numel == 0) {
    if (old_dims == new_dims) {
      new_stride = old_stride;
    } else {
      new_stride = new_dims;
      new_stride[new_dims.size() - 1] = 1;
      for (int i = new_dims.size() - 2; i >= 0; i--) {
        new_stride[i] = new_stride[i + 1] *
                        std::max(static_cast<int64_t>(1), new_dims[i + 1]);
      }
    }
    return true;
  } else {
    int64_t old_numel = 1;
    int64_t new_numel = 1;
    int64_t old_stride_lastvalue = old_stride[old_stride.size() - 1];
    int new_stride_index = new_dims.size() - 1;
    for (int old_dims_index = old_dims.size() - 1; old_dims_index >= 0;
         old_dims_index--) {
      old_numel *= old_dims[old_dims_index];
      if ((old_dims_index == 0) || (old_dims[old_dims_index - 1] != 1 &&
                                    old_stride[old_dims_index - 1] !=
                                        old_numel * old_stride_lastvalue)) {
        while (new_stride_index >= 0 &&
               (new_numel < old_numel || new_dims[new_stride_index] == 1)) {
          new_stride[new_stride_index] = new_numel * old_stride_lastvalue;
          new_numel *= new_dims[new_stride_index];
          new_stride_index--;
        }
        if (new_numel != old_numel) {
          return false;
        }
        if (old_dims_index > 0) {
          old_numel = 1;
          new_numel = 1;
          old_stride_lastvalue = old_stride[old_dims_index - 1];
        }
      }
    }
    if (new_stride_index != -1) {
      return false;
    }
    return true;
  }
  return false;
}

template <typename Context>
void ReshapeStridedKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const IntArray& shape,
                          DenseTensor* out,
                          DenseTensor* xshape UNUSED) {
  MetaTensor meta_out(out);
  InferMetaFromVecValue(x, shape.GetData(), &meta_out);
  DDim stride;
  if (ReshapeStride(x.dims(), x.stride(), out->dims(), stride)) {
    out->set_stride(stride);
    out->ResetHolder(x.Holder());
  } else {
    DenseTensor tmp;
    Contiguous<Context>(dev_ctx, x, &tmp);
    out->set_stride(DenseTensorMeta::calc_stride(out->dims()));
    out->ResetHolder(tmp.Holder());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(reshape,
                                         STRIDED,
                                         phi::ReshapeStridedKernel) {}
