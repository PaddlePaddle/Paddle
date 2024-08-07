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

#include "paddle/phi/kernels/accuracy_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void AccuracyRawKernel(const Context& dev_ctx,
                       const DenseTensor& inference,
                       const DenseTensor& indices,
                       const DenseTensor& label,
                       DenseTensor* accuracy,
                       DenseTensor* correct,
                       DenseTensor* total) {
  int* correct_data = dev_ctx.template Alloc<int>(correct);
  int* total_data = dev_ctx.template Alloc<int>(total);
  float* accuracy_data = dev_ctx.template Alloc<float>(accuracy);

  const int64_t* indices_data = indices.data<int64_t>();
  const int64_t* label_data = label.data<int64_t>();

  PADDLE_ENFORCE_EQ(
      inference.dims().size(),
      2,
      common::errors::InvalidArgument(
          "Rank(Input) of AccuracyOp must be 2, with shape "
          "[sample_number, class_dim], But received rank(Input) is %d",
          inference.dims().size()));

  int64_t num_samples = inference.dims()[0];
  int64_t class_dim = inference.dims()[1];

  int r = xpu::accuracy<int64_t>(dev_ctx.x_context(),
                                 indices_data,
                                 label_data,
                                 num_samples,
                                 class_dim,
                                 correct_data,
                                 total_data,
                                 accuracy_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "accuracy");
}
}  // namespace phi

// TODO(add supported dtype.)
PD_REGISTER_KERNEL(accuracy,
                   XPU,
                   ALL_LAYOUT,
                   phi::AccuracyRawKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
