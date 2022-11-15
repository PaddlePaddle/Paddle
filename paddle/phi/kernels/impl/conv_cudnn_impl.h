// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/phi/kernels/gpudnn/conv_miopen_helper.h"
#else
#include "paddle/phi/kernels/gpudnn/conv_cudnn_v7.h"
#endif

#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/padding.h"

DECLARE_bool(cudnn_deterministic);
DECLARE_int64(conv_workspace_size_limit);
DECLARE_bool(cudnn_exhaustive_search);

namespace phi {

static inline bool IsVoltaOrLater(const phi::GPUContext& dev_ctx) {
  return dev_ctx.GetComputeCapability() >= 70;
}

// inline cudnnTensorFormat_t GetCudnnTensorFormat(
//     const phi::DataLayout& order) {  // Not use
//   switch (order) {
//     case phi::DataLayout::kNHWC:
//       return CUDNN_TENSOR_NHWC;
//     case phi::DataLayout::kNCHW:
//       return CUDNN_TENSOR_NCHW;
//     case phi::DataLayout::NCDHW:
//       return CUDNN_TENSOR_NCHW;  // NOTE: cudnn treat NdTensor as the same
//     case phi::DataLayout::NDHWC:
//       return CUDNN_TENSOR_NHWC;  // add, liyamei
//     default:
//       PADDLE_THROW(phi::errors::Unimplemented(
//           "CUDNN has no equivalent dataLayout for input order."));
//   }
//   return CUDNN_TENSOR_NCHW;
// }

static inline void GetNCDHW(const DDim& dims,
                            const phi::DataLayout& layout,
                            int* N,
                            int* C,
                            int* D,
                            int* H,
                            int* W) {
  *N = dims[0];
  *C = layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == phi::DataLayout::kNCHW ? 0 : 1;
  if (dims.size() == 5) {
    *D = dims[2 - i];
    *H = dims[3 - i];
    *W = dims[4 - i];
  } else {
    *D = 1;
    *H = dims[2 - i];
    *W = dims[3 - i];
  }
}

}  // namespace phi

// PD_REGISTER_KERNEL(convdnn, GPU, ALL_LAYOUT, phi::ConvKernel, float, double
// ) {}
