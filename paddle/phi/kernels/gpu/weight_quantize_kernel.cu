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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/weight_quantize_kernel_gpu_impl.h"

namespace phi {

template <typename T, typename Context>
void WeightQuantizeKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::string& algo,
                          const int32_t arch,
                          DenseTensor* out,
                          DenseTensor* scale) {
  DenseTensor quanted_x;
  dev_ctx.template Alloc<int8_t>(out);
  dev_ctx.template Alloc<T>(scale);
  size_t m = x.dims()[0];
  size_t n = x.dims()[1];
  quanted_x.Resize({static_cast<int64_t>(m), static_cast<int64_t>(n)});
  dev_ctx.template Alloc<int8_t>(&quanted_x);
  std::vector<int> weight_shape{static_cast<int>(x.dims()[0]),
                                static_cast<int>(x.dims()[1])};
  PADDLE_ENFORCE_EQ(
      ((arch == 80) || (arch == 86) || (arch == 75) || (arch == 70)),
      true,
      phi::errors::InvalidArgument(
          "Currently, arch only support 70, 75, 80, 86."));

  if (algo == "llm.int8") {
    std::vector<int> axis = {1, 0};
    funcs::Transpose<Context, int8_t, 2> trans;
    weight_quant_gpu<T, Context>(dev_ctx,
                                 x.data<T>(),
                                 quanted_x.data<int8_t>(),
                                 scale->data<T>(),
                                 weight_shape);
    trans(dev_ctx, quanted_x, out, axis);
  } else if (algo == "weight_only_int8") {
    weight_quant_gpu<T, Context>(dev_ctx,
                                 x.data<T>(),
                                 quanted_x.data<int8_t>(),
                                 scale->data<T>(),
                                 weight_shape);
    weight_permute_gpu<Context>(dev_ctx,
                                quanted_x.data<int8_t>(),
                                out->data<int8_t>(),
                                weight_shape,
                                arch);
  } else if (algo == "weight_only_int4") {
    phi::errors::Unimplemented(
        "Weight quant gpu kernel currently don't support weight_only_int4 "
        "algo, please use cpu version.");
  } else {
    phi::errors::Unimplemented(
        "The algo must be in ['weight_only_int8', 'weight_only_int4', "
        "'llm.int8'], but got[%s]",
        algo);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_quantize,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightQuantizeKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
