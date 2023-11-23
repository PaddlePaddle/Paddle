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

#include "paddle/phi/kernels/array_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/stack_kernel.h"

namespace phi {
template <typename T, typename Context>
void CreateArrayKernel(const Context& dev_ctx,
                       DataType dtype,
                       TensorArray* out) {}

template <typename T, typename Context>
void ArrayLengthKernel(const Context& dev_ctx,
                       const TensorArray& x,
                       DenseTensor* out) {
  out->Resize({1});
  dev_ctx.template Alloc<int64_t>(out);
  *out->data<int64_t>() = static_cast<int64_t>(x.size());
}

template <typename T, typename Context>
void ArrayReadKernel(const Context& dev_ctx,
                     const TensorArray& array,
                     const Scalar& i,
                     DenseTensor* out) {
  size_t offset = i.to<int64_t>();
  PADDLE_ENFORCE_EQ(
      offset < array.size(),
      true,
      errors::InvalidArgument(
          "index %d exceed array size %d.", offset, array.size()));
  phi::Copy(dev_ctx, array[offset], dev_ctx.GetPlace(), false, out);
  out->set_lod(array[offset].lod());
}

template <typename T, typename Context>
void ArrayWriteKernel(const Context& dev_ctx,
                      const TensorArray& array,
                      const DenseTensor& x,
                      const Scalar& i,
                      TensorArray* out) {
  size_t offset = i.to<int64_t>();
  if (offset >= out->size()) {
    out->resize(offset + 1);
  }
  auto* out_tensor = &out->at(offset);
  out_tensor->set_lod(x.lod());
  if (x.memory_size() > 0) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out_tensor);
  }
}

template <typename T, typename Context>
void ArrayToTensorKernel(const Context& dev_ctx,
                         const TensorArray& x,
                         int axis,
                         bool use_stack,
                         DenseTensor* out,
                         DenseTensor* out_index) {
  const size_t n = x.size();
  PADDLE_ENFORCE_GT(
      n,
      0,
      platform::errors::InvalidArgument("Input tensorarray size should > 0,"
                                        "but the received is %d",
                                        n));

  auto out_dims = x[0].dims();
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == static_cast<size_t>(axis)) {
        out_dims[axis] += x[i].dims()[static_cast<int>(j)];
      }
    }
  }
  auto vec = phi::vectorize<int>(out_dims);
  vec.insert(vec.begin() + axis, x.size());  // NOLINT
  out.Resize(phi::make_ddim(vec));

  std::vector<DenseTensor> tmp_tensors(x.size());
  std::vector<const DenseTensor*> inputs;
  for (size_t i = 0; i < x.size(); i++) {
    tmp_tensors[i].ShareDataWith(x[i]);
    inputs.push_back(&tmp_tensors[i]);
  }

  if (use_stack) {
    StackKernel<T, Context>(dev_ctx, inputs, axis, out);
  } else {
    ConcatKernel<T, Context>(dev_ctx, inputs, axis, out);
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(create_array,
                   CPU,
                   ALL_LAYOUT,
                   phi::CreateArrayKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(create_array,
                   GPU,
                   ALL_LAYOUT,
                   phi::CreateArrayKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

PD_REGISTER_KERNEL(array_length,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayLengthKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(array_read,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayReadKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(array_read,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArrayReadKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

PD_REGISTER_KERNEL(array_write,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayWriteKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(array_write,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArrayWriteKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif

PD_REGISTER_KERNEL(array_to_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayToTensorKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(array_to_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArrayToTensorKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
#endif
