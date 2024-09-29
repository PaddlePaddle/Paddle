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

#include "paddle/phi/kernels/accuracy_kernel.h"

#include <algorithm>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void AccuracyKernel(const Context& dev_ctx,
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

  size_t num_samples = inference.dims()[0];
  size_t class_dim = inference.dims()[1];
  *accuracy_data = 0.0f;

  PADDLE_ENFORCE_GT(label.dims().size(),
                    0,
                    common::errors::InvalidArgument(
                        "Rank(Label) of AccuracyOp must greater than 0, "
                        "But received rank(Label) is %d",
                        label.dims().size()));

  PADDLE_ENFORCE_GE(label.dims()[0],
                    inference.dims()[0],
                    common::errors::InvalidArgument(
                        "num_samples(%d) of Label should less than "
                        "or equal to num_samples(%d) of Input",
                        label.dims()[0],
                        num_samples));

  if (num_samples == 0) {
    return;
  }

  int num_correct = 0;
  // assume inference is already the topk of the output
  for (size_t i = 0; i < num_samples; ++i) {
    PADDLE_ENFORCE_GE(
        label_data[i],
        0,
        common::errors::InvalidArgument(
            "label of AccuracyOp must >= 0, But received label[%d] is %d",
            i,
            label_data[i]));
    for (size_t j = 0; j < class_dim; ++j) {
      if (indices_data[i * class_dim + j] == label_data[i]) {
        ++num_correct;
        break;
      }
    }
  }

  *correct_data = num_correct;
  *total_data = static_cast<int>(num_samples);
  *accuracy_data =
      static_cast<float>(num_correct) / static_cast<float>(num_samples);
}
}  // namespace phi

// TODO(add supported dtype.)
PD_REGISTER_KERNEL(
    accuracy, CPU, ALL_LAYOUT, phi::AccuracyKernel, float, double) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
