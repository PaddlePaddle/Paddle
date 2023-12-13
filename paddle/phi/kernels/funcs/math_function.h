/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <cmath>
#include <memory>
#include <vector>

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/data_type.h"
#ifdef PADDLE_WITH_XPU
#include <type_traits>
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#endif

namespace phi {
namespace funcs {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
void BatchTranspose(T* output,
                    const T* input,
                    int64_t batch,
                    int64_t m,
                    int64_t n,
                    const phi::GPUContext* dev_ctx);
#endif
template <typename DeviceContext, typename T>
struct TransposeNormal {
  // for dims >= 7 situation
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& in,
                  phi::DenseTensor* out,
                  const std::vector<int>& axis);
};

template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& in,
                  phi::DenseTensor* out,
                  const std::vector<int>& axis);
};

template <typename DeviceContext, typename T>
struct SetConstant {
  void operator()(const DeviceContext& context,
                  phi::DenseTensor* tensor,
                  T num);
};

template <typename Place>
void set_constant_with_place(const phi::DeviceContext& context,
                             phi::DenseTensor* tensor,
                             const void* value);

void set_constant(const phi::DeviceContext& context,
                  phi::DenseTensor* tensor,
                  const void* value);

template <typename T>
void set_constant(const phi::DeviceContext& context,
                  phi::DenseTensor* tensor,
                  const T value) {
  set_constant(context, tensor, reinterpret_cast<const void*>(&value));
}

template <typename DeviceContext, typename T>
struct RowwiseAdd {
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& vec,
                  phi::DenseTensor* output);
};

template <typename DeviceContext, typename T>
struct ColwiseSum {
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseSum {
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseMean {
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* vec);
};

template <typename Context, typename T>
inline void TransCompute(const int dim,
                         const Context& dev_ctx,
                         const DenseTensor& in,
                         DenseTensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 1:
      Transpose<Context, T, 1> trans1;
      trans1(dev_ctx, in, out, axis);
      break;
    case 2:
      Transpose<Context, T, 2> trans2;
      trans2(dev_ctx, in, out, axis);
      break;
    case 3:
      Transpose<Context, T, 3> trans3;
      trans3(dev_ctx, in, out, axis);
      break;
    case 4:
      Transpose<Context, T, 4> trans4;
      trans4(dev_ctx, in, out, axis);
      break;
    case 5:
      Transpose<Context, T, 5> trans5;
      trans5(dev_ctx, in, out, axis);
      break;
    case 6:
      Transpose<Context, T, 6> trans6;
      trans6(dev_ctx, in, out, axis);
      break;
    default:
      // for dim >= 7 situation
      TransposeNormal<Context, T> trans_normal;
      trans_normal(dev_ctx, in, out, axis);
  }
}

}  // namespace funcs
}  // namespace phi
