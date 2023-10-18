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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/impl/weight_quantize_kernel_gpu_impl.h"
#include "paddle/phi/core/dense_tensor.h"



namespace phi {

template <typename T, typename Context>
void WeightQuantizeKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::string& algo,
                          DenseTensor* out,
                          DenseTensor* scale) { 
  DenseTensor quanted_x; 
  quanted_x.Resize({x.dims()});
  dev_ctx.template Alloc<int8_t>(&quanted_x);
  dev_ctx.template Alloc<int8_t>(out);
  dev_ctx.template Alloc<T>(scale);  
  std::cout<<"#### xdims: ";
  for(int i=0;i<x.dims().size();++i){
    std::cout<<x.dims()[i]<<" ";
  }  
  std::cout<<std::endl; 
  std::vector<int> weight_shape{(int)x.dims()[0], (int)x.dims()[1]};
  if (algo == "weight_only_int8" || algo == "llm.int8") {
    weight_quant_gpu<T, Context>(dev_ctx, 
                                 x.data<T>(),  
                                 quanted_x.data<int8_t>(), 
                                 scale->data<T>(),
                                 weight_shape);
    weight_permute_gpu<Context>(dev_ctx, 
                                quanted_x.data<int8_t>(), 
                                out->data<int8_t>(),
                                weight_shape);
  } else if (algo == "weight_only_int4") {
    // quant_compute<Context, T, int8_t, 4>(dev_ctx, x, out, scale, algo); 
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
