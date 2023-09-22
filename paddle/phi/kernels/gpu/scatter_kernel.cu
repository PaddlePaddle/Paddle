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

#include "paddle/phi/kernels/scatter_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/utils/flags.h"

PD_DECLARE_bool(cudnn_deterministic);

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void index_reduce_cuda_kernel(const T* input,
                                         const IndexT* index,
                                         const T* add_value,
                                         int64_t N,
                                         int64_t stride,
                                         int64_t size,
                                         int64_t delta,
                                         T* output) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, int64_t) {
    int64_t pre_idx = idx / (stride * size);
    int64_t dim_idx = idx % (stride * size) / stride;
    IndexT src_dim_idx = index[dim_idx];
    int64_t input_idx =
        idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
    phi::CudaAtomicAdd(&output[input_idx], add_value[idx]);
  }
}

template <typename T, typename Context>
void IndexReduceKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       const DenseTensor& add_value,
                       int axis,
                       const std::string& reduce,
                       bool include_self,
                       DenseTensor* output) {
  auto input_dim = x.dims();
  auto output_dim = output->dims();
  auto add_value_dim = add_value.dims();

  const auto& index_type = index.dtype();

  int dim = axis;
  dim = dim >= 0 ? dim : dim + input_dim.size();

  auto stride_dim = phi::stride(input_dim);
  int64_t stride = stride_dim[dim];
  int64_t size = add_value_dim[dim];
  int64_t delta = input_dim[dim] - size;

  auto* in_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(output);
  auto* add_value_data = add_value.data<T>();

  int64_t numel = add_value.numel();
  if (numel == 0) {
    return;
  }

  auto stream = ctx.stream();

  unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
  dim3 grid_dim = dim3((numel + block_dim - 1) / block_dim);
  phi::backends::gpu::LimitGridDim(ctx, &grid_dim);

  // copy input to output.
  // todo(@limin29): inplace do not need copy.
  phi::Copy(ctx, x, ctx.GetPlace(), false, output);

  if (FLAGS_cudnn_deterministic) {
    VLOG(2) << "Run grad kernel of index_add with single thread.";
    block_dim = 1;
    grid_dim.x = 1;
  }

  if (index_type == phi::DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    index_reduce_cuda_kernel<T, int64_t>
        <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                             index_data,
                                             add_value_data,
                                             numel,
                                             stride,
                                             size,
                                             delta,
                                             out_data);
  } else {
    const int* index_data = index.data<int>();
    index_reduce_cuda_kernel<T, int>
        <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                             index_data,
                                             add_value_data,
                                             numel,
                                             stride,
                                             size,
                                             delta,
                                             out_data);
  }
}

template <typename T, typename Context>
void ScatterKernel(const Context& ctx,
                   const DenseTensor& x,
                   const DenseTensor& index,
                   const DenseTensor& updates,
                   bool overwrite,
                   int axis,
                   const std::string& reduce,
                   bool include_self,
                   DenseTensor* out) {
  auto index_type = index.dtype();
  PADDLE_ENFORCE_EQ(
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64,
      true,
      phi::errors::InvalidArgument(
          "scatter_op Index holds the wrong type, it holds [%s],"
          "but desires to be [%s] or [%s].",
          index_type,
          phi::DataType::INT32,
          phi::DataType::INT64));

  PADDLE_ENFORCE_EQ(
      reduce == "add" || reduce == "mul" || reduce == "muliply" ||
          reduce == "mean" || reduce == "amin" || reduce == "amax",
      true,
      phi::errors::InvalidArgument(
          "Reduce holds the wrong value, it holds [%s],"
          "but desires to be add, mul, multiply, mean, amin, amax.",
          reduce));

  std::string reducer = reduce;
  if (overwrite) {
    reducer = "assign";
  }

  IndexReduceKernel<T, Context>(
      ctx, x, index, updates, axis, reducer, include_self, out);

  // phi::Copy(ctx, x, ctx.GetPlace(), false, out);
  // int reduce_type = reduce == "sum" ? 0 : 1;
  // if (index_type == phi::DataType::INT32) {
  //   phi::funcs::GPUScatterAssign<T, int32_t>(
  //       ctx, updates, index, out, overwrite, reduce_type);
  // } else {
  //   phi::funcs::GPUScatterAssign<T, int64_t>(
  //       ctx, updates, index, out, overwrite, reduce_type);
  // }
}

}  // namespace phi

PD_REGISTER_KERNEL(scatter,
                   GPU,
                   ALL_LAYOUT,
                   phi::ScatterKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
