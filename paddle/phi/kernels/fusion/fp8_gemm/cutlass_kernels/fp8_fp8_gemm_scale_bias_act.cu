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

#include "fp8_fp8_gemm_scale_bias_act.h"

#include "gemm_scale.h"
#include "gemm_scale_bias.h"
#include "gemm_scale_bias_act.h"

namespace phi{
namespace fusion{
namespace cutlass_internal{

bool fp8_fp8_gemm_scale_bias_act(GemmEpilogueAllParams params){
    if((params.input_dtype == "e4m3")&&(params.output_dtype == "bf16")){
      if(!params.bias && (params.activation_type == "identity" || params.activation_type == "")){
        dispatch_gemm_scale<phi::dtype::float8_e4m3fn, phi::dtype::bfloat16>(params);
      }else if(params.bias && (params.activation_type == "identity" || params.activation_type == "")){
        dispatch_gemm_scale_bias<phi::dtype::float8_e4m3fn, phi::dtype::bfloat16>(params);
      }else{
        dispatch_gemm_scale_bias_act<phi::dtype::float8_e4m3fn, phi::dtype::bfloat16>(params);
      }
    }else if((params.input_dtype == "e4m3")&&(params.output_dtype == "fp16")){
      if(!params.bias && (params.activation_type == "identity" || params.activation_type == "")){
        dispatch_gemm_scale<phi::dtype::float8_e4m3fn, phi::dtype::float16>(params);
      }else if(params.bias && (params.activation_type == "identity" || params.activation_type == "")){

  std::cout<<"params.bias: "  << params.bias <<std::endl;

        dispatch_gemm_scale_bias<phi::dtype::float8_e4m3fn, phi::dtype::float16>(params);
      }else{
        dispatch_gemm_scale_bias_act<phi::dtype::float8_e4m3fn, phi::dtype::float16>(params);
      }
    }else if((params.input_dtype == "e5m2")&&(params.output_dtype == "bf16")){
      if(!params.bias && (params.activation_type == "identity" || params.activation_type == "")){
        dispatch_gemm_scale<phi::dtype::float8_e5m2, phi::dtype::bfloat16>(params);
      }else if(params.bias && (params.activation_type == "identity" || params.activation_type == "")){
        dispatch_gemm_scale_bias<phi::dtype::float8_e5m2, phi::dtype::bfloat16>(params);
      }else{
        dispatch_gemm_scale_bias_act<phi::dtype::float8_e5m2, phi::dtype::bfloat16>(params);
      }
    }else if((params.input_dtype == "e5m2")&&(params.output_dtype == "fp16")){
      if(!params.bias && (params.activation_type == "identity" || params.activation_type == "")){
        dispatch_gemm_scale<phi::dtype::float8_e5m2, phi::dtype::float16>(params);
      }else if(params.bias && (params.activation_type == "identity" || params.activation_type == "")){
        dispatch_gemm_scale_bias<phi::dtype::float8_e5m2, phi::dtype::float16>(params);
      }else{
        dispatch_gemm_scale_bias_act<phi::dtype::float8_e5m2, phi::dtype::float16>(params);
      }
    }
    return false;
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
