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

#include "paddle/infrt/kernel/phi/dense_tensor_kernels.h"
#include "paddle/infrt/dialect/phi/data_type.h"
#include "paddle/infrt/kernel/phi/context_kernels.h"

namespace infrt {
namespace kernel {
namespace phi {

::phi::DenseTensor CreateDenseTensor(
    const ::phi::CPUContext& context,
    host_context::Attribute<std::vector<int64_t>> dims,
    host_context::Attribute<std::vector<int64_t>> lod,
    host_context::Attribute<::infrt::LayoutType> layout,
    host_context::Attribute<::infrt::PrecisionType> precision) {
  return ::phi::DenseTensor(
      const_cast<::phi::Allocator*>(&context.GetAllocator()),
      ::phi::DenseTensorMeta(cvtPrecision2Phi(precision.get()),
                             ::phi::make_ddim(dims.get()),
                             cvtLayout2Phi(layout.get()),
                             {}));
}

void FillDenseTensorF32(::phi::DenseTensor* dense_tensor,
                        host_context::Attribute<std::vector<float>> value) {
  auto place = ::phi::CPUPlace();
  float* a_data = dense_tensor->mutable_data<float>(place);
  for (int64_t i = 0; i < dense_tensor->numel(); ++i) {
    a_data[i] = (value.get())[i];
  }
}

void PrintDenseTensor(::phi::DenseTensor* dense_tensor) {
#define PRINT_META_DATA(PHI_DATATYPE, DTYPE)              \
  case ::phi::DataType::PHI_DATATYPE: {                   \
    DTYPE* data = dense_tensor->data<DTYPE>();            \
    if (dense_tensor->numel() == 0) break;                \
    std::cout << data[0];                                 \
    for (int64_t i = 1; i < dense_tensor->numel(); i++) { \
      std::cout << "," << data[i];                        \
    }                                                     \
    break;                                                \
  }

  ::phi::DDim dims = dense_tensor->dims();
  std::cout << "dense_tensor: shape=shape" << dims.to_str() << ","
            << " value=[";
  switch (dense_tensor->dtype()) {
    PRINT_META_DATA(FLOAT32, float);
    PRINT_META_DATA(INT32, int32_t);
    default:
      std::cout << "Error! Unsupported data type!\n";
  }
  std::cout << "]\n";
#undef PRINT_META_DATA
}
}  // namespace phi
}  // namespace kernel
}  // namespace infrt
