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
#include <map>

#include "dual_gemm_scale_bias_swiglu.h"       // NOLINT
#include "dual_gemm_scale_swiglu.h"            // NOLINT
#include "fp8_fp8_dual_gemm_scale_bias_act.h"  // NOLINT

namespace phi {
namespace fusion {
namespace cutlass_internal {

std::map<std::string, int> config_map1{
    {"e4m3_e4m3_swiglu", 0},
    {"e4m3_e4m3_bias0_bias1_swiglu", 1},
    {"e4m3_e4m3_geglu", 2},
    {"e4m3_e4m3_bias0_bias1_geglu", 3},
};

bool fp8_fp8_dual_gemm_scale_bias_act(DualGemmEpilogueAllParams params) {
  std::cout << "params.gemm_config:" << params.gemm_config << std::endl;
  std::cout << "config_map[&params.gemm_config()] :"
            << config_map1[params.gemm_config] << std::endl;

  switch (config_map1[params.gemm_config]) {
    case 0:
      dispatch_dual_gemm_scale_swiglu<phi::dtype::float8_e4m3fn,
                                      phi::dtype::float8_e4m3fn>(params);
      break;
    case 1:
      dispatch_dual_gemm_scale_bias_swiglu<phi::dtype::float8_e4m3fn,
                                           phi::dtype::float8_e4m3fn>(params);
      break;
    // case 2:
    //   dispatch_dual_gemm_scale_bias_swiglu<phi::dtype::float8_e4m3fn,
    //                                 phi::dtype::float8_e4m3fn>(params);
    //   break;
    // case 3:
    //   dispatch_dual_gemm_scale_geglu<phi::dtype::float8_e4m3fn,
    //   phi::dtype::float8_e4m3fn>(
    //       params);
    //   break;
    // case 5:
    //   dispatch_dual_gemm_scale_geglu<phi::dtype::float8_e4m3fn,
    //   phi::dtype::float8_e4m3fn>(
    //       params);
    //   break;
    // case 6:
    //   dispatch_dual_gemm_scale_bias_geglu<phi::dtype::float8_e4m3fn,
    //   phi::dtype::float8_e4m3fn>(
    //       params);
    //   break;
    // case 7:
    //   dispatch_dual_gemm_scale_bias_geglu<phi::dtype::float8_e4m3fn,
    //                                 phi::dtype::float8_e4m3fn>(params);
    //   break;
    default:
      throw std::runtime_error("fp8_fp8_fp8_gemm_fused Config is invalid.");
      break;
  }
  return false;
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
