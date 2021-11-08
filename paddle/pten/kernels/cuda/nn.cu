// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/cuda/funcs/elementwise/elementwise.h"
#include "paddle/pten/kernels/cuda/nn.h"
#include "paddle/pten/kernels/functions/general/elementwise_functor.h"

namespace pten {

template <typename T>
void ElementwiseAdd(const CUDAContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  std::vector<DenseTensor*> outputs;
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
      dev_ctx, inputs, &outputs, axis, general::AddFunctor<T>());
}

}  // namespace pten

PT_REGISTER_MODULE(NnCUDA);

using float16 = ::paddle::platform::float16;
using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL("elementwise_add",
                   CUDA,
                   ANY,
                   pten::ElementwiseAdd,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
