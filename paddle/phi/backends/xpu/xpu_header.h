/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#ifdef PADDLE_WITH_XPU_BKCL
#include "xpu/bkcl.h"
#endif
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"
#ifdef PADDLE_WITH_XPU_PLUGIN
#include "xpu/plugin.h"
#endif
#ifdef PADDLE_WITH_XPU_FFT
#include "fft/cuComplex.h"
#endif
namespace xpu = baidu::xpu::api;

template <typename T>
class XPUTypeTrait {
 public:
  using Type = T;
};

template <>
class XPUTypeTrait<phi::dtype::float16> {
 public:
  using Type = float16;
};

template <>
class XPUTypeTrait<phi::dtype::bfloat16> {
 public:
  using Type = bfloat16;
};

template <typename T>
class XPUTypeToPhiType {
 public:
  using Type = T;
};

template <>
class XPUTypeToPhiType<float16> {
 public:
  using Type = phi::dtype::float16;
};

template <>
class XPUTypeToPhiType<bfloat16> {
 public:
  using Type = phi::dtype::bfloat16;
};

// XPUCopyTypeTrait is the same as XPUTypeTrait except for double, int16_t, and
// uint8_t. Used for ops that simply copy data and do not need to calculate
template <typename T>
class XPUCopyTypeTrait {
 public:
  using Type = T;
};

template <>
class XPUCopyTypeTrait<phi::dtype::float16> {
 public:
  using Type = float16;
};

template <>
class XPUCopyTypeTrait<phi::dtype::bfloat16> {
 public:
  using Type = bfloat16;
};

template <>
class XPUCopyTypeTrait<double> {
 public:
  using Type = int64_t;
};

template <>
class XPUCopyTypeTrait<int16_t> {
 public:
  using Type = float16;
};

template <>
class XPUCopyTypeTrait<uint8_t> {
 public:
  using Type = int8_t;
};

// Reverse the given `axes` of a contiguous row-major buffer `x` into `y`,
// which must not alias `x`. `xpu::flip` leaves parts of its output
// uninitialized for some shapes (observed nondeterministically on (8, 6)
// fp32), while the negative-step path of `xpu::strided_slice` -- the same
// lowering every `x[::-1]`-style slice takes -- is long covered by XPU CI.
// `strided_slice` is not instantiated for 1-byte types, so those keep
// `xpu::flip`.
template <typename T>
inline int XPUReverseAxes(xpu::Context* ctx,
                          const T* x,
                          T* y,
                          const std::vector<int64_t>& shape,
                          const std::vector<int64_t>& axes) {
  if constexpr (sizeof(T) > 1) {
    std::vector<int64_t> starts(shape.size(), 0);
    std::vector<int64_t> ends(shape);
    std::vector<int64_t> steps(shape.size(), 1);
    for (int64_t axis : axes) {
      starts[axis] = shape[axis] - 1;
      ends[axis] = -1;  // with a negative step, -1 runs down to index 0
      steps[axis] = -1;
    }
    return xpu::strided_slice<T>(ctx, x, y, shape, starts, ends, steps);
  } else {
    return xpu::flip<T>(ctx, x, y, shape, axes);
  }
}

#ifdef PADDLE_WITH_XPU_FFT
template <typename T>
class XPUComplexTypeTrait {
 public:
  using Type = T;
};

template <>
class XPUComplexTypeTrait<float> {
 public:
  using Type = cuFloatComplex;
};

template <>
class XPUComplexTypeTrait<double> {
 public:
  using Type = cuDoubleComplex;
};
#endif
