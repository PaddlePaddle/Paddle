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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/kernels/index_add_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/cpu/index_add_impl.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void IndexAddKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    // const DenseTensor& index,
                    const IntArray& index,
                    int axis,
                    // float added_value,
                    const Scalar& add_value,
                    DenseTensor* output) {
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, output);
  if (axis < 0) {
    axis += x.dims().size();
  }
  // const auto& index_type = index.dtype();

  // bool index_type_match =
  //     index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  // PADDLE_ENFORCE_EQ(index_type_match,
  //                   true,
  //                   phi::errors::InvalidArgument(
  //                       "Input(Index) holds the wrong type, it holds %s, but "
  //                       "desires to be %s or %s",
  //                       index_type,
  //                       phi::DataType::INT32,
  //                       phi::DataType::INT64));

  // auto added_val = static_cast<T>(added_value);
  auto add_val = add_value.to<T>();


  // if (index_type == phi::DataType::INT32) {
  //   IndexAddInner<Context, T, int>(dev_ctx, index, output, axis, added_val);
  // } else if (index_type == phi::DataType::INT64) {
  //   IndexAddInner<Context, T, int64_t>(dev_ctx, index, output, axis, added_val);
  // }
  IndexAddCPUImpl<Context, T>(dev_ctx, index, output, axis, add_val);

}

}  // namespace phi

PD_REGISTER_KERNEL(index_add,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexAddKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {}
