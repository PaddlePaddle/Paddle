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

#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/framework/data_type.h"
#include "paddle/phi/core/ddim.h"
#define WARP_SIZE 32
#define BLOCK_SIZE 1024

namespace phi {

// shared memory
template <typename T>
struct SharedMemory {
  __device__ T* getPointer() { return NULL; }
};

template <>
struct SharedMemory<phi::dtype::float16> {
  __device__ phi::dtype::float16* getPointer() {
    extern __shared__ phi::dtype::float16 s_fp16[];
    return s_fp16;
  }
};

template <>
struct SharedMemory<float> {
  __device__ float* getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <>
struct SharedMemory<double> {
  __device__ double* getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

template <>
struct SharedMemory<int32_t> {
  __device__ int32_t* getPointer() {
    extern __shared__ int32_t s_int32_t[];
    return s_int32_t;
  }
};

template <>
struct SharedMemory<int64_t> {
  __device__ int64_t* getPointer() {
    extern __shared__ int64_t s_int64_t[];
    return s_int64_t;
  }
};

template <>
struct SharedMemory<int16_t> {
  __device__ int16_t* getPointer() {
    extern __shared__ int16_t s_int16_t[];
    return s_int16_t;
  }
};

template <>
struct SharedMemory<int8_t> {
  __device__ int8_t* getPointer() {
    extern __shared__ int8_t s_int8_t[];
    return s_int8_t;
  }
};

template <typename T>
struct CompMinOp {
  __device__ __forceinline__ T initial() {
    return static_cast<T>(std::numeric_limits<T>::max());
  }
  __device__ __forceinline__ bool operator()(const T a, const T b) const {
    return b < a;
  }
};

template <typename T>
struct CompMaxOp {
  __device__ __forceinline__ T initial() {
    return static_cast<T>(std::numeric_limits<T>::lowest());
  }
  __device__ __forceinline__ bool operator()(const T a, const T b) const {
    return a < b;
  }
};

template <typename T, typename IndType, typename CompOp>
__device__ void ArgWraper(T* values,
                          unsigned int* idx,
                          CompOp comp_op,
                          const unsigned int res_diff) {
  for (int stride = WARP_SIZE; stride > 0; stride >>= 1) {
    if (stride < res_diff &&
        comp_op(values[threadIdx.x], values[threadIdx.x + stride])) {
      values[threadIdx.x] = values[threadIdx.x + stride];
      idx[threadIdx.x] = idx[threadIdx.x + stride];
    }
  }
}

template <typename T, typename IndType, typename CompOp>
__global__ void ArgCudaKernel(const unsigned int length,
                              T* d_values,
                              IndType* d_index,
                              IndType* out,
                              const int64_t out_offset,
                              CompOp comp_op,
                              bool is_init) {
  unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int border = length >> 1;
  SharedMemory<T> shared;
  T* s_arg_values = shared.getPointer();
  s_arg_values[threadIdx.x] = comp_op.initial();
  if (tidx > border) return;

  const unsigned int res_diff = length - tidx;
  unsigned int* s_arg_idx = (unsigned int*)&s_arg_values[blockDim.x];
  unsigned int arg_id = is_init ? tidx : d_index[tidx];
  T arg_value = d_values[tidx];

  unsigned compare_idx = border + tidx;
  if (border < res_diff and comp_op(arg_value, d_values[compare_idx])) {
    arg_id = compare_idx;
    arg_value = d_values[arg_id];
  }

  s_arg_values[threadIdx.x] = arg_value;
  s_arg_idx[threadIdx.x] = arg_id;

  for (border = blockDim.x >> 1; border > 32; border >>= 1) {
    if (threadIdx.x > border) return;

    __syncthreads();
    compare_idx = threadIdx.x + border;  // within this block
    if (border < res_diff && compare_idx < blockDim.x &&
        comp_op(arg_value, s_arg_values[compare_idx])) {
      arg_value = s_arg_values[compare_idx];
      arg_id = s_arg_idx[compare_idx];
    }

    s_arg_values[threadIdx.x] = arg_value;
    s_arg_idx[threadIdx.x] = arg_id;
  }

  if (threadIdx.x < 32)
    ArgWraper<T, CompOp>(s_arg_values, s_arg_idx, comp_op, res_diff);

  if (threadIdx.x == 0) {
    d_values[blockIdx.x] = s_arg_values[0];
    d_index[blockIdx.x] = static_cast<IndType>(s_arg_idx[0]);
    out[out_offset] = d_index[blockIdx.x];
  }
}

template <typename Context, typename T, typename IndType, typename CompOp>
void ArgCUDAImpl(const Context& dev_ctx,
                 T* in,
                 int64_t length,
                 IndType* out,
                 CompOp comp_op,
                 const int64_t out_offset) {  //--->h_index (arg_max)
  int grid_size = std::ceil(length / float(BLOCK_SIZE) / 2);

  DenseTensor d_index;  // init d_index
  d_index.Resize(phi::make_ddim({grid_size}));
  dev_ctx.template Alloc<IndType>(&d_index);
  IndType* d_index_ptr = d_index.data<IndType>();

  const unsigned int s_mem_size =
      BLOCK_SIZE * (sizeof(T) + sizeof(unsigned int));
  ArgCudaKernel<T, IndType, CompOp><<<grid_size, BLOCK_SIZE, s_mem_size>>>(
      length, in, d_index_ptr, out, out_offset, comp_op, true);

  while (grid_size > 1) {
    length = grid_size;
    grid_size = std::ceil(length / float(BLOCK_SIZE) / 2);
    ArgCudaKernel<T, IndType, CompOp><<<grid_size, BLOCK_SIZE, s_mem_size>>>(
        length, in, d_index_ptr, out, out_offset, comp_op, false);
  }
}

template <typename Context, typename T, typename CompOp>
struct VisitDataCudaArgMinMaxFunctor {
  const Context& dev_ctx;
  const DenseTensor& x;
  int64_t axis;
  bool keepdims;
  bool flatten;
  CompOp comp_op;
  DenseTensor* out;

  explicit VisitDataCudaArgMinMaxFunctor(const Context& dev_ctx,
                                         const DenseTensor& x,
                                         int64_t axis,
                                         bool keepdims,
                                         bool flatten,
                                         CompOp comp_op,
                                         DenseTensor* out)
      : dev_ctx(dev_ctx),
        x(x),
        axis(axis),
        keepdims(keepdims),
        flatten(flatten),
        comp_op(comp_op),
        out(out) {}

  template <typename IndType>
  void apply() const {
    phi::DDim x_dims = x.dims();
    auto rank = x_dims.size();

    DenseTensor input;
    IndType* out_ptr = dev_ctx.template Alloc<IndType>(out);

    if (flatten || rank == 1) {
      input.Resize(phi::make_ddim({x.numel()}));
      T* input_ptr = dev_ctx.template Alloc<T>(&input);
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, &input);
      ArgCUDAImpl<Context, T, IndType, CompOp>(
          dev_ctx, input_ptr, x.numel(), out_ptr, comp_op, 0);
    } else {
      int new_axis = axis;
      if (axis < 0) new_axis += rank;

      std::vector<int> perm;
      int64_t pre_dim = 1;
      DDim permed_shape(x_dims);
      for (int64_t i = 0; i < rank; i++) {
        if (i != axis) {
          perm.push_back(i);
          permed_shape[i] = x_dims[i];
          pre_dim *= x_dims[i];
        }
      }
      perm.push_back(new_axis);
      permed_shape[rank - 1] = x_dims[new_axis];
      int64_t post_dim = x_dims[new_axis];

      input.Resize(permed_shape);
      T* input_ptr = dev_ctx.template Alloc<T>(&input);
      funcs::TransCompute<Context, T>(rank, dev_ctx, x, &input, perm);

      int grid_size = std::ceil(post_dim / float(BLOCK_SIZE) / 2);
      DenseTensor d_index;  // init d_index
      d_index.Resize(phi::make_ddim({grid_size}));
      IndType* d_index_ptr = dev_ctx.template Alloc<IndType>(&d_index);
      const unsigned int s_mem_size =
          BLOCK_SIZE * (sizeof(T) + sizeof(unsigned int));

      for (int64_t i = 0; i < pre_dim; i++) {
        int64_t pos = static_cast<int64_t>(i * post_dim);
        ArgCUDAImpl<Context, T, IndType, CompOp>(
            dev_ctx, input_ptr + pos, post_dim, out_ptr, comp_op, i);
      }
    }
    out->Resize(out->dims());
  }
};

template <typename Context, typename T, typename CompOp>
void ArgMinMaxOpCUDAKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const Scalar& axis,
                           bool keepdims,
                           bool flatten,
                           int dtype,
                           CompOp comp_op,
                           DenseTensor* out) {
  if (dtype < 0) {
    paddle::framework::VisitDataTypeTiny(
        static_cast<paddle::framework::proto::VarType::Type>(
            paddle::framework::proto::VarType::INT64),
        VisitDataCudaArgMinMaxFunctor<Context, T, CompOp>(
            dev_ctx, x, axis.to<int64_t>(), keepdims, flatten, comp_op, out));
    return;
  }
  paddle::framework::VisitDataTypeTiny(
      static_cast<paddle::framework::proto::VarType::Type>(dtype),
      VisitDataCudaArgMinMaxFunctor<Context, T, CompOp>(
          dev_ctx, x, axis.to<int64_t>(), keepdims, flatten, comp_op, out));
}

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  DenseTensor* out) {
  ArgMinMaxOpCUDAKernel<Context, T, CompMinOp<T>>(
      dev_ctx, x, axis, keepdims, flatten, dtype, CompMinOp<T>(), out);
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  int dtype,
                  DenseTensor* out) {
  ArgMinMaxOpCUDAKernel<Context, T, CompMaxOp<T>>(
      dev_ctx, x, axis, keepdims, flatten, dtype, CompMaxOp<T>(), out);
}

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
