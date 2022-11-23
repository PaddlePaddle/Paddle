/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/full_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename InT, typename OutT = InT>
<<<<<<< HEAD
struct FullFuctor {
  OutT value;

  template <typename VType>
  explicit inline FullFuctor(VType val) {
=======
struct FullFunctor {
  OutT value;

  template <typename VType>
  explicit inline FullFunctor(VType val) {
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    value = static_cast<OutT>(val);
  }

  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(value);
  }
};

template <typename T, typename Context>
<<<<<<< HEAD
void CooFullLikeKernel(const Context& dev_ctx,
=======
void FullLikeCooKernel(const Context& dev_ctx,
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                       const SparseCooTensor& x,
                       const Scalar& val,
                       DataType dtype,
                       SparseCooTensor* out) {
<<<<<<< HEAD
  phi::Copy<Context>(dev_ctx,
                     x.non_zero_indices(),
                     dev_ctx.GetPlace(),
                     false,
                     out->mutable_non_zero_indices());

  DenseTensor* values = out->mutable_non_zero_elements();
  values->Resize(x.non_zero_elements().dims());
=======
  phi::Copy<Context>(
      dev_ctx, x.indices(), dev_ctx.GetPlace(), false, out->mutable_indices());

  DenseTensor* values = out->mutable_values();
  values->Resize(x.values().dims());
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  dev_ctx.template Alloc<T>(values);

  std::vector<const DenseTensor*> inputs = {};
  std::vector<DenseTensor*> outputs = {values};
  int numel = values->numel();
  if (numel > 0) {
    phi::funcs::ElementwiseKernel<T>(
<<<<<<< HEAD
        dev_ctx, inputs, &outputs, FullFuctor<T>(val.to<T>()));
=======
        dev_ctx, inputs, &outputs, FullFunctor<T>(val.to<T>()));
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  }
  out->set_dims(x.dims());
}

template <typename T, typename Context>
<<<<<<< HEAD
void CsrFullLikeKernel(const Context& dev_ctx,
=======
void FullLikeCsrKernel(const Context& dev_ctx,
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                       const SparseCsrTensor& x,
                       const Scalar& val,
                       DataType dtype,
                       SparseCsrTensor* out) {
<<<<<<< HEAD
  phi::Copy<Context>(dev_ctx,
                     x.non_zero_crows(),
                     dev_ctx.GetPlace(),
                     false,
                     out->mutable_non_zero_crows());

  phi::Copy<Context>(dev_ctx,
                     x.non_zero_cols(),
                     dev_ctx.GetPlace(),
                     false,
                     out->mutable_non_zero_cols());

  DenseTensor* values = out->mutable_non_zero_elements();
  values->Resize(x.non_zero_elements().dims());
=======
  phi::Copy<Context>(
      dev_ctx, x.crows(), dev_ctx.GetPlace(), false, out->mutable_crows());

  phi::Copy<Context>(
      dev_ctx, x.cols(), dev_ctx.GetPlace(), false, out->mutable_cols());

  DenseTensor* values = out->mutable_values();
  values->Resize(x.values().dims());
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  dev_ctx.template Alloc<T>(values);

  std::vector<const DenseTensor*> inputs = {};
  std::vector<DenseTensor*> outputs = {values};
  int numel = values->numel();
  if (numel > 0) {
    phi::funcs::ElementwiseKernel<T>(
<<<<<<< HEAD
        dev_ctx, inputs, &outputs, FullFuctor<T>(val.to<T>()));
=======
        dev_ctx, inputs, &outputs, FullFunctor<T>(val.to<T>()));
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
  }
  out->set_dims(x.dims());
}

}  // namespace phi

<<<<<<< HEAD
PD_REGISTER_KERNEL(coo_full_like,
                   GPU,
                   ALL_LAYOUT,
                   phi::CooFullLikeKernel,
=======
PD_REGISTER_KERNEL(full_like_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::FullLikeCooKernel,
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

<<<<<<< HEAD
PD_REGISTER_KERNEL(csr_full_like,
                   GPU,
                   ALL_LAYOUT,
                   phi::CsrFullLikeKernel,
=======
PD_REGISTER_KERNEL(full_like_csr,
                   GPU,
                   ALL_LAYOUT,
                   phi::FullLikeCsrKernel,
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                   float,
                   double,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
