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

#include "paddle/phi/kernels/arg_min_max_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/reduce.h"

#if defined(__NVCC__) || defined(__HIPCC__)

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include <limits>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/phi/core/ddim.h"

namespace phi {

namespace {  // NOLINT
template <typename K, typename V>
using KeyValuePair = cub::KeyValuePair<K, V>;
using paddle::platform::FastDivMod;

}  // end namespace

#define FIXED_BLOCK_DIM_CASE_BASE(log2_block_dim, ...)  \
  case (1 << (log2_block_dim)): {                       \
    constexpr auto kBlockDim = (1 << (log2_block_dim)); \
    __VA_ARGS__;                                        \
  } break

#define FIXED_BLOCK_DIM_CASE(...)               \
  FIXED_BLOCK_DIM_CASE_BASE(10, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(9, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(8, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(7, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(6, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(5, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(4, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(3, ##__VA_ARGS__);

template <typename T, typename IndType, class Reducer, size_t BlockDim>
__global__ void ArgCUDAKernel(const int64_t height,     // n * h
                              const int64_t width,      // c
                              const int64_t post_size,  // h
                              const FastDivMod div_mod,
                              const Reducer reducer,
                              const T init,
                              const T* in,
                              IndType* out) {
  typedef cub::BlockReduce<KeyValuePair<int, T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int idx = blockIdx.x; idx < height; idx += gridDim.x) {
    KeyValuePair<int, T> kv_pair = {-1, init};
    auto divmod = div_mod.Divmod(idx);
    int h = divmod.val[0];
    int w = divmod.val[1];

    for (int k = threadIdx.x; k < width; k += blockDim.x) {
      kv_pair =
          reducer({k, in[h * width * post_size + k * post_size + w]}, kv_pair);
    }
    kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, reducer);
    if (threadIdx.x == 0) {
      out[idx] = static_cast<IndType>(kv_pair.key);
    }
    __syncthreads();
  }
}

template <typename T, typename IndType, class Reducer>
void ComputeFullArg(const phi::GPUContext& dev_ctx,
                    const DenseTensor& input,
                    DenseTensor* indices,
                    const int64_t pre,
                    const int64_t post,
                    const int64_t n) {
  auto cu_stream = dev_ctx.stream();
  auto ComputeBlockSize = [](int64_t col) { return 512; };

  int64_t max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int64_t height = pre * post;
  int64_t width = n;
  int64_t grid_size = height < max_grid_dimx ? height : max_grid_dimx;

  const T* in_data = input.data<T>();
  IndType* out_data = dev_ctx.template Alloc<IndType>(indices);

  // post size==> divisor
  FastDivMod post_divmod = paddle::platform::FastDivMod(post);
  const size_t kBlockDim = 512;
  if (typeid(Reducer) == typeid(cub::ArgMax)) {
    ArgCUDAKernel<T, IndType, Reducer, kBlockDim>
        <<<grid_size, kBlockDim, 0, cu_stream>>>(
            height,
            width,
            post,
            post_divmod,
            Reducer(),
            std::numeric_limits<T>::lowest(),
            in_data,
            out_data);

  } else {
    ArgCUDAKernel<T, IndType, Reducer, kBlockDim>
        <<<grid_size, kBlockDim, 0, cu_stream>>>(height,
                                                 width,
                                                 post,
                                                 post_divmod,
                                                 Reducer(),
                                                 std::numeric_limits<T>::max(),
                                                 in_data,
                                                 out_data);
  }
}

template <typename Context, typename T, class Reducer>
struct VisitDataCudaArgMinMaxFunctor {
  const Context& dev_ctx;
  const DenseTensor& x;
  int64_t axis;
  bool keepdims;
  bool flatten;
  DenseTensor* out;

  explicit VisitDataCudaArgMinMaxFunctor(const Context& dev_ctx,
                                         const DenseTensor& x,
                                         int64_t axis,
                                         bool keepdims,
                                         bool flatten,
                                         DenseTensor* out)
      : dev_ctx(dev_ctx),
        x(x),
        axis(axis),
        keepdims(keepdims),
        flatten(flatten),
        out(out) {}

  template <typename IndType>
  void apply() const {
    /*
    phi::DDim x_dims;
    int new_axis = axis;
    if (flatten) {
      x_dims = phi::make_ddim({x.numel()});
      // if flatten, the axis just as 0
      new_axis = 0;
    } else {
      x_dims = x.dims();
      if (axis < 0) new_axis = axis + x.dims().size();
    }

    int64_t numel = x.numel();
    int64_t groups = numel / x_dims[new_axis];
    int64_t pre = 1;
    int64_t post = 1;
    int64_t n = x_dims[new_axis];

    for (int i = 0; i < new_axis; i++) {
      pre *= x_dims[i];
    }

    for (int i = new_axis + 1; i < x_dims.size(); i++) {
      post *= x_dims[i];
    }

    ComputeFullArg<T, IndType, Reducer>(dev_ctx, x, out, pre, post, n);*/

    phi::funcs::ReduceKernel<T, T, kps::MaxFunctor, kps::IdentityFunctor<T>>(
        dev_ctx, x, out, kps::IdentityFunctor<T>(), {axis.to<int>()});
  }
};

template <typename Context, typename T, class Reducer>
void ArgMinMaxOpCUDAKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const Scalar& axis,
                           bool keepdims,
                           bool flatten,
                           int dtype,
                           DenseTensor* out) {
  if (dtype < 0) {
    paddle::framework::VisitDataTypeTiny(
        static_cast<paddle::framework::proto::VarType::Type>(
            paddle::framework::proto::VarType::INT64),
        VisitDataCudaArgMinMaxFunctor<Context, T, Reducer>(
            dev_ctx, x, axis.to<int64_t>(), keepdims, flatten, out));
    return;
  }
  paddle::framework::VisitDataTypeTiny(
      static_cast<paddle::framework::proto::VarType::Type>(dtype),
      VisitDataCudaArgMinMaxFunctor<Context, T, Reducer>(
          dev_ctx, x, axis.to<int64_t>(), keepdims, flatten, out));
}

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  DenseTensor* out) {
  ArgMinMaxOpCUDAKernel<Context, T, kps::ArgMinFunctor>
      dev_ctx, x, axis, keepdims, flatten, dtype, out);
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  DenseTensor* out) {
  ArgMinMaxOpCUDAKernel<Context, T, cub::ArgMax>(
      dev_ctx, x, axis, keepdims, flatten, dtype, out);
}

#endif

}  // namespace phi

PD_REGISTER_KERNEL(arg_min,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArgMinKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   int16_t,
                   uint8_t) {}

PD_REGISTER_KERNEL(arg_max,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArgMaxKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   int16_t,
                   uint8_t) {}
