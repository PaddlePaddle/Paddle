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

#include "paddle/phi/kernels/gather_nd_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/tile_kernel.h"

namespace phi {

template <typename T, typename Context>
void GatherNdKernel(const Context &dev_ctx,
                    const DenseTensor &x,
                    const DenseTensor &index,
                    DenseTensor *out) {
  dev_ctx.template Alloc<T>(out);
  if (x.numel() == 0 || out->numel() == 0) return;
  // The result dims is
  //   Index.shape[:-1] + X.shape[Index.shape[-1]:]
  // If the last dimension of index is 0, set it to 1 and tile x.
  auto index_dims = index.dims();
  std::vector<int64_t> out_dims;
  if (index_dims[index_dims.size() - 1] == 0) {
    for (int i = 0; i < index_dims.size() - 1; ++i) {
      out_dims.emplace_back(index_dims[i]);
    }
    for (int i = 0; i < x.dims().size(); ++i) {
      out_dims.emplace_back(1);
    }
    phi::TileKernel<T, Context>(dev_ctx, x, phi::IntArray(out_dims), out);
    return;
  }
  if (index.dims()[0] == 0 && index.numel() == 0) return;
  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s] or [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));
  if (index_type == phi::DataType::INT32) {
    phi::funcs::GPUGatherNd<T, int>(dev_ctx, x, index, out);
  } else if (index_type == phi::DataType::INT64) {
    phi::funcs::GPUGatherNd<T, int64_t>(dev_ctx, x, index, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gather_nd,
                   GPU,
                   ALL_LAYOUT,
                   phi::GatherNdKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
