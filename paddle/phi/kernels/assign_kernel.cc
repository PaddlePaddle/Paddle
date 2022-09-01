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

#include "paddle/phi/kernels/assign_kernel.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename Context>
void AssignKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) {
  paddle::framework::TensorCopy(x, x.place(), out);
}

template <typename Context>
void AssignRawKernel(const Context& dev_ctx,
                     const paddle::optional<DenseTensor>& x,
                     DenseTensor* out) {
  if (x) {
    if (!x->IsInitialized()) {
      return;
    }
    auto& x_tensor = *x.get_ptr();
    AssignKernel<Context>(dev_ctx, x_tensor, out);
  }
}

// Note: use `const paddle::optional<std::vector<const DenseTensor*>&> x`
// as input if needed
template <typename Context>
void AssignArrayKernel(const Context& dev_ctx,
                       const TensorArray& x,
                       TensorArray* out) {
  for (size_t i = 0; i < x.size(); ++i) {
    AssignKernel<Context>(dev_ctx, x[i], &out->at(i));
  }
}

template <typename T, typename Context>
typename std::enable_if<std::is_same<T, bool>::value>::type CopyVectorToTensor(
    const Context& dev_ctx,
    const std::vector<Scalar>& values,
    DenseTensor* out) {
  // If attribute value dtype is vector<bool>, it will be converted to
  // vector<int>. at the same time, we can not use vector<bool> to hold
  // the value, because the c++ use bit value to replace byte value.
  std::vector<int> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.emplace_back(val.to<int>());
  }
  paddle::framework::TensorFromVector(assign_values, dev_ctx, out);

  // use the array to replace to vector
  bool* array_ptr = new T[assign_values.size()];
  for (unsigned int i = 0; i < assign_values.size(); i++) {
    array_ptr[i] = static_cast<T>(assign_values[i]);
  }
  paddle::framework::TensorFromArray(
      array_ptr, assign_values.size(), dev_ctx, out);
  delete[] array_ptr;
}

template <typename T, typename Context>
typename std::enable_if<!std::is_same<T, bool>::value>::type CopyVectorToTensor(
    const Context& dev_ctx,
    const std::vector<Scalar>& values,
    DenseTensor* out) {
  std::vector<T> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.emplace_back(val.to<T>());
  }
  paddle::framework::TensorFromVector(assign_values, dev_ctx, out);
}

template <typename T, typename Context>
void AssignValueKernel(const Context& dev_ctx,
                       const std::vector<int>& shape,
                       DataType dtype,
                       const std::vector<Scalar>& values,
                       DenseTensor* out) {
  auto template_dtype = paddle::experimental::CppTypeToDataType<T>::Type();
  PADDLE_ENFORCE_EQ(
      dtype,
      template_dtype,
      phi::errors::InvalidArgument("Argument dtype mismatch for kernel dtype, "
                                   "argument dtype is %s, kernel dtype is %s.",
                                   dtype,
                                   template_dtype));
  CopyVectorToTensor<T>(dev_ctx, values, out);
  out->Resize(phi::make_ddim(shape));
}

}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(
    assign, CPU, ALL_LAYOUT, phi::AssignKernel<phi::CPUContext>, ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(assign_raw,
                           CPU,
                           ALL_LAYOUT,
                           phi::AssignRawKernel<phi::CPUContext>,
                           ALL_DTYPE) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_GENERAL_KERNEL(assign_array,
                           CPU,
                           ALL_LAYOUT,
                           phi::AssignArrayKernel<phi::CPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_KERNEL(assign_value,
                   CPU,
                   ALL_LAYOUT,
                   phi::AssignValueKernel,
                   bool,
                   int,
                   float,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(
    assign, GPU, ALL_LAYOUT, phi::AssignKernel<phi::GPUContext>, ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(assign_raw,
                           GPU,
                           ALL_LAYOUT,
                           phi::AssignRawKernel<phi::GPUContext>,
                           ALL_DTYPE) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_GENERAL_KERNEL(assign_array,
                           GPU,
                           ALL_LAYOUT,
                           phi::AssignArrayKernel<phi::GPUContext>,
                           ALL_DTYPE) {}
PD_REGISTER_KERNEL(assign_value,
                   GPU,
                   ALL_LAYOUT,
                   phi::AssignValueKernel,
                   bool,
                   int,
                   float,
                   int64_t) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_GENERAL_KERNEL(
    assign, XPU, ALL_LAYOUT, phi::AssignKernel<phi::XPUContext>, ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(assign_raw,
                           XPU,
                           ALL_LAYOUT,
                           phi::AssignRawKernel<phi::XPUContext>,
                           ALL_DTYPE) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
PD_REGISTER_GENERAL_KERNEL(assign_array,
                           XPU,
                           ALL_LAYOUT,
                           phi::AssignArrayKernel<phi::XPUContext>,
                           ALL_DTYPE) {}
#endif
