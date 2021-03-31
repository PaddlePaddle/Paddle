// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_fp16.h>
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#endif

namespace paddle {
namespace operators {

/*
template <template <int i> typename func, int end, int current = 0>
template <typename... Args>
static inline void with_args(Args &&... args) {
  func<current>::apply(std::forward<Args>(args)...);
  static_unroll<func, end, current + 1>::with_args(args...);
}
};
*/

template <template <int i> typename func, int end>
struct static_unroll<func, end, end> {
  template <typename... Args>
  static inline void with_args(Args... args) {}
};

// template <typename T, int vec_size>
// struct alignas(sizeof(T) * vec_size) aligned_vector {
//   T scalar_array[vec_size];
// };

template <typename DeviceContext, typename T, typename Functor>
class ElementwiseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_tensors = ctx.MultiInput<framework::LoDTensor>("Input");
    auto *out_tensor = ctx.Output<framework::LoDTensor>("Out");
    bool is_broadcast;

    using vec_input_t = std::vector<const framework::Tensor *>;
    vec_input_t in_array;
    in_array.reserve(in_tensors.size());
    for (auto *in_tensor : in_tensors) {
      is_broadcast = in_tensor->dims() == out_tensor->dims();
      in_array.emplace_back(*in_tensor);
    }

    if (!is_broadcast && (in_tensors.size() == 1)) {
      SameDimsElemwise<DeviceContext, T, Functor, vec_input_t>(ctx, &in_array,
                                                               out_tensor);
    } else {
      BroadcastElementwise<DeviceContext, T, Functor, vec_input_t>(
          ctx, &in_array, out_tensor);
    }
  }
};

// template <typename T, typename Functor, typename inp_calc_t, int N, int
// nDims>
// __global__ void ElementwiseKernel(vec_input_t ins, out, numel, ) {}

// template <template <int i> typename func, int end, int current = 0>
// template <typename... Args>
// static inline void with_args(Args &&... args) {
//   func<current>::apply(std::forward<Args>(args)...);
//   static_unroll<func, end, current + 1>::with_args(args...);
// }
// };

// template <template <int i> typename func, int end>
// struct static_unroll<func, end, end> {
//   template <typename... Args>
//   static inline void with_args(Args... args) {}
// };

// template <int N>
// }
// ;

template <typename args_t>
void load(args_t *args, int idx) {
  constexpr int arity = std::tuple_size<args_t>::value;
  detail::static_unroll<detail::vectorized_load_helper, MAX_DIMS>::with_args(
      *this, args, idx);
}

template <typename DeviceContext, typename T, typename Functor,
          typename vec_input_t>
void BroadcastElementwise(const framework::ExecutionContext &ctx,
                          vec_input_t *ins, framework::Tensor *out) {
  auto input_num = ins->size();
  size_t numel = out->numel();
  switch (input_num) {
    case 2: {
      auto input_calc = OffsetCalculator<vec_input_t, 2>(ins, out);
      BroadcastElementwiseKernel<T, Functor, input_calc, 2>(
          ins, out, numel);  // 用元模板展开!
      break;
    }
    case 3: {
      auto input_calc = OffsetCalculator<vec_input_t, 3>(ins, out);
      BroadcastElementwiseKernel<T, Functor, input_calc, 3>(
          ins, out, numel);  // 用元模板展开!
      break;
    }
    default: {
      auto input_calc = OffsetCalculator<vec_input_t>(ins, out);
      BroadcastElementwiseKernel<T, Functor, input_calc>(ins, out, numel);
    }
  }
}

/*
* Dimension Merging took account of below conditions:
* 1. [32, 2, 16, 12] + [32, 1,  1, 12] => [32, 32,  12] + [32, 1,  12]
* 2. [32, 2, 16, 12] + [32, 1, 16, 12] => [32,  2, 192] + [32, 1, 192]
* 3. [32, 2, 16, 12] + [ 1, 2, 16, 12] => [32, 192] + [ 1, 192]
* 4. [32, 2, 16, 12] + [32, 1,  1,  1] => [32, 192] + [32,   1]
* 5. [32, 2, 16, 12] + [32, 2 ]        => [64, 192] + [64, 1]
* 6. [32, 2, 16, 12] + [16, 12]        => [32, 2, 192] + [1, 1, 192]
* 7. [32, 2, 16, 12] + [32]            => [32, 384] + [32, 1]
* 8. [32, 2, 16, 12] + [2]             => [32, 2, 192] + [1, 2, 1]
* 9. [32, 2,  1,  1] + [1,  1, 16, 12] => [32, 2,   1] + [1, 1, 192]
*10. [32, 1,  1,  1] + [1,  2, 16, 12] => [32, 1] + [1, 192]
*11. [32, 1, 16, 12] + [32, 2,  1,  1] => [32, 1, 192] + [32, 2, 1]
*12. [32, 1, 16,  1] + [1,  2, 16, 12] => No support
*13. [32, 1, 16,  1] + [1,  2,  1, 12] => No support
* 先进行维度补充, 再进行维度合并,
*/
template <int N, typename vec_input_t>
void MergeDims(vec_input_t *ins, framework::DDim *out_dims) {
  PADDLE_ENFORCE_GE(
      out_dims->size(), MAX_DIMS,
      platform::errors::InvalidArgument(
          "Output tensor`s dim is %d, bigger than upper limitation %d\n",
          paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
          paddle::framework::DataTypeToString(
              framework::proto::VarType::INT32)));

  for (*in_tensor : ins) {
    PADDLE_ENFORCE_GE(
        in_tensor->dims().size(), MAX_DIMS,
        platform::errors::InvalidArgument(
            "Input tensor`s dim is %d, bigger than upper limitation %d\n",
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32),
            paddle::framework::DataTypeToString(
                framework::proto::VarType::INT32)));
    if (in_tensor->dims() ==) {
      12
    }
  }
}

// template <int N, typename vec_input_t>  // N : input tensors.
// static OffsetCalculator<N, vec_input_t> innput_offset_calculator(
//     vec_input_t *ins, framework::Tensor *out) {
//   constexpr int input_num = std::max<int>(N, 1);
//   std::array<int, input_num> shift_array;
//   std::array<int, input_num> mul_array;

//   std::reverse(out->dims().begin(), out->dims().end());
//   for (*in : ins) {
//     std::reverse(in->dims().begin(), in->dims().end());
//   }
//   MergeDims(ins, out->dims())
// }

// template<typename func_t, typename array_t, typename inp_calc_t, typename
// out_calc_t, typename loader_t, typename storer_t>
// __global__ void BroadcastElementwiseKernel(int N, func_t f, array_t data,
//                                             inp_calc_t ic, out_calc_t oc,
//                                             loader_t l, storer_t s)
// {
//   int remaining = N - block_work_size * blockIdx.x;
//   auto policy = memory::policies::unroll<array_t, inp_calc_t, out_calc_t,
//   loader_t, storer_t>(data, remaining, ic, oc, l, s);
//   ElementwiseCalculator(f, policy);
// }

template <typename T, int N, int nDims>
__device__ ElementwiseCalculatorCore() {
  for (int i = 0; i < N; ++i) {
    // __device__ load<T, nDims>;
  }

  // __device__ func<T>;

  // __device__ store<T>;
}

}  // namespace operators
}  // namespace paddle
