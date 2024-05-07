// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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

#include "paddle/common/flags.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/cuda/cudnn_workspace_helper.h"
/**
 * CUDNN related FLAG
 * Name: FLAGS_conv_workspace_size_limit
 * Since Version: 0.13.0
 * Value Range: uint64, default=512 (MB)
 * Example:
 * Note: The internal function of cuDNN obtains the fastest matching algorithm
 *       within this memory limit. Usually, faster algorithms can be chosen in
 *       larger workspaces, but memory space can also be significantly
 * increased.
 *       Users need to balance memory and speed.
 */
PHI_DEFINE_EXPORTED_int64(conv_workspace_size_limit,
                          phi::backends::gpu::kDefaultConvWorkspaceSizeLimitMB,
                          "cuDNN convolution workspace limit in MB unit.");
#endif

/**
 * The fmha mode in FusedMultiTransformer
 * Value Range: string, {naive, cutlass, flash_attention_v2}
 */
PHI_DEFINE_EXPORTED_string(fmha_mode,
                           "cutlass",
                           "The mode of fmha in FusedMultiTransformer.");

PHI_DEFINE_EXPORTED_bool(print_matrix, false, "");
PHI_DEFINE_EXPORTED_bool(fuse_softmax, false, "");

PHI_DEFINE_EXPORTED_int64(custom_allreduce_one_shot_threshold,
                          -1,
                          "");  // 393216
PHI_DEFINE_EXPORTED_int64(custom_allreduce_two_shot_threshold,
                          -1,
                          "");  // 50331648
