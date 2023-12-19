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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature FusedMatmulOpArgumentMapping(
    const ArgumentMappingContext& ctx UNUSED) {
  return KernelSignature("fused_matmul",
                         {"X", "Y", "ResidualData"},
                         {"trans_x",
                          "trans_y",
                          "matmul_alpha",
                          "fuse_activation",
                          "fuse_alpha",
                          "fuse_beta",
                          "fused_output_scale",
                          "fused_reshape_X",
                          "fused_transpose_X",
                          "fused_reshape_Y",
                          "fused_transpose_Y",
                          "fused_reshape_Out",
                          "fused_transpose_Out",
                          "mkldnn_data_type",
                          "Scale_x",
                          "Scale_y",
                          "Scale_in_eltwise",
                          "Scale_out",
                          "force_fp32_output"},
                         {"Out"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(fused_matmul, phi::FusedMatmulOpArgumentMapping);
