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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/conv_cudnn_grad_kernel.h"
#include "paddle/pten/kernels/conv_cudnn_kernel.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/fluid/framework/eigen.h"
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/operators/conv_miopen_helper.h"
#else
#include "paddle/fluid/operators/conv_cudnn_helper.h"
#endif

#include "paddle/fluid/operators/math/padding.h"
#include "paddle/fluid/platform/cudnn_workspace_helper.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/profiler.h"

#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/pten/kernels/cpu/conv_util.h"
#include "paddle/pten/kernels/funcs/batch_norm_utils.h"

DECLARE_bool(cudnn_deterministic);
DECLARE_uint64(conv_workspace_size_limit);
DECLARE_bool(cudnn_exhaustive_search);

namespace pten {

static inline bool IsVoltaOrLater(const pten::GPUContext& dev_ctx) {
  return dev_ctx.GetComputeCapability() >= 70;
}

inline cudnnTensorFormat_t GetCudnnTensorFormat(
    const pten::DataLayout& order) {  // Not use
  switch (order) {
    case pten::DataLayout::kNHWC:
      return CUDNN_TENSOR_NHWC;
    case pten::DataLayout::kNCHW:
      return CUDNN_TENSOR_NCHW;
    case pten::DataLayout::NCDHW:
      return CUDNN_TENSOR_NCHW;  // NOTE: cudnn treat NdTensor as the same
    case pten::DataLayout::NDHWC:
      return CUDNN_TENSOR_NHWC;  // add, liyamei
    default:
      PADDLE_THROW(pten::errors::Unimplemented(
          "CUDNN has no equivalent dataLayout for input order."));
  }
  return CUDNN_TENSOR_NCHW;
}

static inline void GetNCDHW(const paddle::framework::DDim& dims,
                            const pten::DataLayout& layout,
                            int* N,
                            int* C,
                            int* D,
                            int* H,
                            int* W) {
  *N = dims[0];
  *C = layout == pten::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
  int i = layout == pten::DataLayout::kNCHW ? 0 : 1;
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

}  // namespace pten

// PT_REGISTER_KERNEL(convdnn, GPU, ALL_LAYOUT, pten::ConvKernel, float, double
// ) {}
