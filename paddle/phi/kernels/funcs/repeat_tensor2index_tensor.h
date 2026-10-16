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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/expand_kernel.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/xpu_context.h"
#endif

namespace phi {
namespace funcs {

// Broadcast a length-1 `repeats` tensor to `axis_len` so a single repeat value
// applies to every element along the axis, matching torch.repeat_interleave.
// Returns `repeats` unchanged when no broadcast is needed; otherwise fills
// `holder` and returns it.
template <typename Context>
const DenseTensor &BroadcastRepeatsIfNeeded(const Context &ctx,
                                            const DenseTensor &repeats,
                                            int64_t axis_len,
                                            DenseTensor *holder) {
  if (repeats.dims()[0] != 1 || axis_len == 1) {
    return repeats;
  }
  if (axis_len == 0) {
    holder->Resize({0});
    if (repeats.dtype() == DataType::INT32) {
      ctx.template Alloc<int>(holder);
    } else {
      ctx.template Alloc<int64_t>(holder);
    }
    return *holder;
  }
  if (repeats.dtype() == DataType::INT32) {
    ExpandKernel<int, Context>(ctx, repeats, IntArray({axis_len}), holder);
  } else {
    ExpandKernel<int64_t, Context>(ctx, repeats, IntArray({axis_len}), holder);
  }
  return *holder;
}

template <typename Context, typename RepeatsT>
class RepeatsTensor2IndexTensorFunctor {
 public:
  void operator()(const Context &ctx,
                  const DenseTensor &repeats,
                  DenseTensor *index);
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename RepeatsT>
class RepeatsTensor2IndexTensorFunctor<GPUContext, RepeatsT> {
 public:
  void operator()(const GPUContext &ctx,
                  const DenseTensor &repeats,
                  DenseTensor *index);
};
#else
template <typename RepeatsT>
class RepeatsTensor2IndexTensorFunctor<CPUContext, RepeatsT> {
 public:
  void operator()(const CPUContext &ctx,
                  const DenseTensor &repeats,
                  DenseTensor *index);
};
#endif

#ifdef PADDLE_WITH_XPU
template <typename RepeatsT>
class RepeatsTensor2IndexTensorFunctor<XPUContext, RepeatsT> {
 public:
  void operator()(const XPUContext &ctx,
                  const DenseTensor &repeats,
                  DenseTensor *index);
};
#endif

}  // namespace funcs
}  // namespace phi
