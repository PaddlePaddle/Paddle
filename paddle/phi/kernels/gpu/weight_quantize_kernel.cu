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
#include "paddle/common/enforce.h"
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
                          const int32_t group_size,
                          DenseTensor* out,
                          DenseTensor* scale) {
  PADDLE_ENFORCE_EQ(
      ((group_size == -1) || (group_size == 64) || (group_size == 128)),
      true,
      common::errors::InvalidArgument(
          "Currently, group_size only support -1(per-channel), 64 or 128."));

  DenseTensor quanted_x;
  dev_ctx.template Alloc<int8_t>(out);
  size_t m = x.dims()[0];
  size_t n = x.dims()[1];
  quanted_x.Resize({static_cast<int64_t>(m), static_cast<int64_t>(n)});
  dev_ctx.template Alloc<int8_t>(&quanted_x);
  std::vector<int> weight_shape{static_cast<int>(x.dims()[0]),
                                static_cast<int>(x.dims()[1])};
#ifndef PADDLE_WITH_HIP
  PADDLE_ENFORCE_EQ(
      ((arch == 70) || (arch == 75) || (arch == 80) || (arch == 86) ||
       (arch == 89) || (arch == 90)),
      true,
      common::errors::InvalidArgument(
          "Currently, arch only support 70, 75, 80, 86, 89, 90."));
#endif
  if (algo == "llm.int8") {
    dev_ctx.template Alloc<float>(scale);
    std::vector<int> axis = {1, 0};
    funcs::Transpose<Context, int8_t, 2> trans;
    weight_quant_gpu<T, Context>(dev_ctx,
                                 x.data<T>(),
                                 quanted_x.data<int8_t>(),
                                 scale->data<float>(),
                                 weight_shape,
                                 arch,
                                 algo);
    trans(dev_ctx, quanted_x, out, axis);
  } else if (algo == "weight_only_int8") {
    dev_ctx.template Alloc<T>(scale);
    weight_quant_gpu<T, Context>(dev_ctx,
                                 x.data<T>(),
                                 quanted_x.data<int8_t>(),
                                 scale->data<T>(),
                                 weight_shape,
                                 arch,
                                 algo);
#ifdef PADDLE_WITH_HIP
    std::vector<int> axis = {1, 0};
    funcs::Transpose<Context, int8_t, 2> trans;
    trans(dev_ctx, quanted_x, out, axis);
#else
    weight_permute_gpu<Context>(dev_ctx,
                                quanted_x.data<int8_t>(),
                                out->data<int8_t>(),
                                weight_shape,
                                arch,
                                algo);
#endif
  } else if (algo == "weight_only_int4") {
#ifdef PADDLE_WITH_HIP
    PADDLE_FATAL(
        "Weight quant gpu kernel currently don't support weight_only_int4 "
        "algo, please use cpu version.");
#else
    dev_ctx.template Alloc<T>(scale);
    weight_quant_gpu<T, Context>(dev_ctx,
                                 x.data<T>(),
                                 quanted_x.data<int8_t>(),
                                 scale->data<T>(),
                                 weight_shape,
                                 arch,
                                 algo);
    weight_permute_gpu<Context>(dev_ctx,
                                quanted_x.data<int8_t>(),
                                out->data<int8_t>(),
                                weight_shape,
                                arch,
                                algo);
#endif
  } else {
    PADDLE_FATAL(
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
