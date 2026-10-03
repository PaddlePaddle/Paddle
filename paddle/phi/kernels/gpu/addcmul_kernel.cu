// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/addcmul_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/addcmul_functor.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void AddcmulKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& tensor1,
                   const DenseTensor& tensor2,
                   const Scalar& value,
                   DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  funcs::AddcmulFunctor<T> functor(funcs::GetAddcmulValue<T>(value));
  if (out->numel() == 0) {
    return;
  }

  // BroadcastKernel aligns all lower-rank inputs with the same axis, which is
  // wrong when the three inputs have different ranks. Extend every input to
  // the output rank first, this only changes the metadata.
  const int rank = out->dims().size();
  DenseTensor input_ext(input);
  DenseTensor tensor1_ext(tensor1);
  DenseTensor tensor2_ext(tensor2);
  input_ext.Resize(funcs::ExtendDims2Rank(input.dims(), rank));
  tensor1_ext.Resize(funcs::ExtendDims2Rank(tensor1.dims(), rank));
  tensor2_ext.Resize(funcs::ExtendDims2Rank(tensor2.dims(), rank));

  std::vector<const DenseTensor*> ins = {
      &input_ext, &tensor1_ext, &tensor2_ext};
  std::vector<DenseTensor*> outs = {out};
  funcs::BroadcastKernel<T>(dev_ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(addcmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddcmulKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {}
