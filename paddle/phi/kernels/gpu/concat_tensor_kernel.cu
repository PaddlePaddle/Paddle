// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/concat_tensor_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {

template <typename T>
bool IsConcated(const std::vector<const DenseTensor*>& tensors) {
  if (tensors.size() < 2) {
    return true;
  }
  std::vector<int64_t> tail_dims(tensors.size());
  std::vector<const T*> tensors_data(tensors.size());

  for (size_t i = 0; i < tensors.size(); i++) {
    auto dims = tensors[i]->dims();
    auto tail_dim = dims[dims.size() - 1];
    tail_dims[i] = tail_dim;
    tensors_data[i] = tensors[i]->data<T>();
  }

  for (size_t i = 0; i < tensors.size() - 1; i++) {
    if (tensors_data[i] + tail_dims[i] != tensors_data[i + 1]) {
      return false;
    }
  }

  return true;
}

template <typename T, typename Context>
void ConcatTensorKernel(const Context& dev_ctx,
                        const std::vector<const DenseTensor*>& x,
                        DenseTensor* concated_out) {
  bool is_concated = IsConcated<T>(x);
  if (is_concated) {
    auto axis_val = phi::funcs::ComputeAxis(-1, x[0]->dims().size());
    std::vector<phi::DDim> x_dims;
    for (size_t i = 0; i < x.size(); ++i) {
      x_dims.push_back(x[i]->dims());
    }
    phi::DDim concated_out_dims =
        phi::funcs::ComputeAndCheckShape(true, x_dims, axis_val);
    concated_out->ShareDataWith(*x[0]);
    concated_out->Resize(concated_out_dims);
  } else {
    ConcatKernel<T, Context>(dev_ctx, x, -1, concated_out);
  }
}
template void ConcatTensorKernel<int, CPUContext>(
    const CPUContext& dev_ctx,
    const std::vector<const DenseTensor*>& x,
    DenseTensor* concated_out);

}  // namespace phi

PD_REGISTER_KERNEL(concat_tensor,
                   GPU,
                   ALL_LAYOUT,
                   phi::ConcatTensorKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
