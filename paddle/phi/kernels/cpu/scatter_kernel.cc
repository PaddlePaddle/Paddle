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

#include "paddle/phi/kernels/scatter_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/cpu/scatter_kernel_impl.h"
#include "paddle/phi/kernels/funcs/scatter.h"

namespace phi {

template <typename T, typename Context>
void ScatterKernel(const Context &ctx,
                   const DenseTensor &x,
                   const DenseTensor &index,
                   const DenseTensor &updates,
                   bool overwrite,
                   int axis,
                   const std::string &reduce,
                   bool include_self,
                   DenseTensor *out) {
  const auto &index_type = index.dtype();
  PADDLE_ENFORCE_EQ(
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s].",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));

  PADDLE_ENFORCE_EQ(
      reduce == "add" || reduce == "mul" || reduce == "muliply" ||
          reduce == "mean" || reduce == "amin" || reduce == "amax",
      true,
      phi::errors::InvalidArgument(
          "Reduce holds the wrong value, it holds [%s],"
          "but desires to be add, mul, multiply, mean, amin, amax.",
          reduce));

  DenseTensor new_index = index;
  DenseTensor new_updates = updates;

  if (new_index.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(
        index.dims()[1],
        1,
        phi::errors::InvalidArgument("index.dims()[1] should be 1 when "
                                     "index.dims().size() =2 in scatter_op."
                                     "But received value is [%d]",
                                     new_index.dims()[1]));
    auto index_dim = new_index.dims()[0];
    new_index.Resize(make_ddim({index_dim}));
  } else if (index.dims().size() == 0) {
    new_index.Resize(make_ddim({1}));

    if (updates.dims().size() == x.dims().size() - 1) {
      auto dims = vectorize(updates.dims());
      dims.insert(dims.begin(), 1);
      new_updates.Resize(make_ddim(dims));
    }
  } else {
    PADDLE_ENFORCE_EQ(
        new_index.dims().size() == 1,
        true,
        phi::errors::InvalidArgument("index.dims().size() should be 1 in "
                                     "scatter_op. But received value is [%d]",
                                     new_index.dims().size()));
  }

  auto src_dims = updates.dims();
  auto dst_dims = out->dims();

  if (new_index.dims().size() != 0) {
    // check src shape and dst shape should match
    for (int i = 1; i < src_dims.size(); i++)
      PADDLE_ENFORCE_EQ(
          src_dims[i],
          dst_dims[i],
          phi::errors::InvalidArgument(
              "The dimensions of the source tensor and target tensor should"
              " match, but received source tensor's %d-th dimension is %d,"
              "target tensor's %d-th dimension is %d.",
              i,
              src_dims[i],
              i,
              dst_dims[i]));
  }

  std::string reducer = reduce;
  if (overwrite) {
    reducer = "assign";
  }

  IndexReduceBaseKernel<T, Context>(
      ctx, x, new_index, new_updates, axis, reducer, include_self, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    scatter, CPU, ALL_LAYOUT, phi::ScatterKernel, float, double, int, int64_t) {
}
