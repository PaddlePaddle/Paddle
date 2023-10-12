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

#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/index_add_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

PD_DECLARE_bool(cudnn_deterministic);

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void index_reduce_cuda_kernel(const T* input,
                                         const IndexT* index,
                                         const T* source,
                                         int reduce,
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
    if (reduce == 0 || reduce == 1) {
      phi::CudaAtomicAdd(&output[input_idx], source[idx]);
    } else if (reduce == 2) {
      phi::CudaAtomicMul(&output[input_idx], source[idx]);
    } else if (reduce == 3) {
      phi::CudaAtomicMin(&output[input_idx], source[idx]);
    } else if (reduce == 4) {
      phi::CudaAtomicMax(&output[input_idx], source[idx]);
    } else if (reduce == 5) {
      output[input_idx] = source[idx];
    }
  }
}

template <typename T, typename IndexT>
__global__ void index_fill_cuda_kernel(const T* input,
                                       const IndexT* index,
                                       const T* source,
                                       T init_val,
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

    output[input_idx] = init_val;
  }
}

template <typename T, typename Context>
void IndexReduceKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       const DenseTensor& source,
                       int axis,
                       const std::string& reduce,
                       bool include_self,
                       DenseTensor* output) {
  auto input_dim = x.dims();
  auto output_dim = output->dims();
  auto source_dim = source.dims();

  const auto& index_type = index.dtype();

  int dim = axis;
  dim = dim >= 0 ? dim : dim + input_dim.size();

  auto stride_dim = phi::stride(input_dim);
  int64_t stride = stride_dim[dim];
  int64_t size = source_dim[dim];
  int64_t delta = input_dim[dim] - size;

  auto* in_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(output);
  auto* source_data = source.data<T>();

  int64_t numel = source.numel();
  if (numel == 0) {
    return;
  }

  auto stream = ctx.stream();

  unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
  dim3 grid_dim = dim3((numel + block_dim - 1) / block_dim);
  phi::backends::gpu::LimitGridDim(ctx, &grid_dim);

  if (FLAGS_cudnn_deterministic) {
    VLOG(2) << "Run grad kernel of index_add with single thread.";
    block_dim = 1;
    grid_dim.x = 1;
  }

  phi::Copy(ctx, x, ctx.GetPlace(), false, output);
  if (!include_self) {
    T init_val;
    if (reduce == "mul" || reduce == "multiply") {
      init_val = static_cast<T>(1);
    } else if (reduce == "amin") {
      init_val = std::numeric_limits<T>::has_infinity
                     ? std::numeric_limits<T>::infinity()
                     : std::numeric_limits<T>::max();
    } else if (reduce == "amax") {
      init_val = std::numeric_limits<T>::has_infinity
                     ? -std::numeric_limits<T>::infinity()
                     : std::numeric_limits<T>::lowest();
    } else {
      init_val = static_cast<T>(0);
    }

    if (index_type == phi::DataType::INT64) {
      const int64_t* index_data = index.data<int64_t>();
      index_fill_cuda_kernel<T, int64_t>
          <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                               index_data,
                                               source_data,
                                               init_val,
                                               numel,
                                               stride,
                                               size,
                                               delta,
                                               out_data);
    } else {
      const int* index_data = index.data<int>();
      index_fill_cuda_kernel<T, int>
          <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                               index_data,
                                               source_data,
                                               init_val,
                                               numel,
                                               stride,
                                               size,
                                               delta,
                                               out_data);
    }
  }

  int reduce_type = 0;
  if (reduce == "add") {
    reduce_type = 0;
  } else if (reduce == "mean") {
    reduce_type = 1;
  } else if (reduce == "mul" || reduce == "multiply") {
    reduce_type = 2;
  } else if (reduce == "amin") {
    reduce_type = 3;
  } else if (reduce == "amax") {
    reduce_type = 4;
  } else if (reduce == "assign") {
    reduce_type = 5;
  }

  if (index_type == phi::DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    index_reduce_cuda_kernel<T, int64_t>
        <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                             index_data,
                                             source_data,
                                             reduce_type,
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
                                             source_data,
                                             reduce_type,
                                             numel,
                                             stride,
                                             size,
                                             delta,
                                             out_data);
  }

  if (reduce == "mean") {
    auto zeros = Full<T, Context>(ctx, vectorize(input_dim), 0);
    auto ones = Full<T, Context>(ctx, vectorize(input_dim), 1);
    auto counts = include_self ? ones : zeros;
    auto src_ones = Full<T, Context>(ctx, vectorize(source.dims()), 1);
    auto src_cnts = IndexAdd<T, Context>(ctx, counts, index, src_ones, dim);
    auto mask = Equal<T, Context>(ctx, src_cnts, zeros);

    auto src_cnts_wo_zeros = Where<T, Context>(ctx, mask, ones, src_cnts);
    auto out = Divide<T, Context>(ctx, *output, src_cnts_wo_zeros);
    phi::Copy(ctx, out, ctx.GetPlace(), false, output);
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

  DenseTensor new_index = index;

  if (index.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(
        index.dims()[1],
        1,
        phi::errors::InvalidArgument("index.dims()[1] should be 1 when "
                                     "index.dims().size() =2 in scatter_op."
                                     "But received value is [%d]",
                                     index.dims()[1]));
    auto index_dim = new_index.dims()[0];
    new_index.Resize(make_ddim({index_dim}));
  } else if (index.dims().size() == 0) {
    new_index.Resize(make_ddim({1}));
  } else {
    PADDLE_ENFORCE_EQ(
        index.dims().size() == 1,
        true,
        phi::errors::InvalidArgument("index.dims().size() should be 1 in "
                                     "scatter_op. But received value is [%d]",
                                     index.dims().size()));
  }

  auto src_dims = updates.dims();
  auto dst_dims = out->dims();

  if (new_index.dims().size() != 0) {
    // check src shape and dst shape should match
    for (int i = 1; i < src_dims.size(); i++)
      PADDLE_ENFORCE_EQ(
          src_dims[i],
          dst_dims[i],
          phi::errors::InvalidArgument(
              "The dimensions of the source tensor and target tensor should"
              " match, but received source tensor's %d-th dimension is %d,"
              "target tensor's %d-th dimension is %d.",
              i,
              src_dims[i],
              i,
              dst_dims[i]));
  }

  auto input_dim = x.dims();
  axis = axis >= 0 ? axis : axis + input_dim.size();
  int index_size = new_index.dims().size();

  DenseTensor index_cpu;
  phi::Copy(ctx, new_index, phi::CPUPlace(), false, &index_cpu);

  for (int i = 0; i < index_size; i++) {
    if (index_type == phi::DataType::INT32) {
      const int* index_data = index_cpu.data<int>();

      PADDLE_ENFORCE_GE(
          index_data[i],
          0,
          phi::errors::InvalidArgument(
              "Variable value (index) of OP(index_add) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              input_dim[axis],
              index_data[i]));
      PADDLE_ENFORCE_LT(
          index_data[i],
          input_dim[axis],
          phi::errors::InvalidArgument(
              "Variable value (index) of OP(index_add) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              input_dim[axis],
              index_data[i]));

    } else if (index_type == phi::DataType::INT64) {
      const int64_t* index_data = index_cpu.data<int64_t>();

      PADDLE_ENFORCE_GE(
          index_data[i],
          0,
          phi::errors::InvalidArgument(
              "Variable value (index) of OP(index_add) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              input_dim[axis],
              index_data[i]));
      PADDLE_ENFORCE_LT(
          index_data[i],
          input_dim[axis],
          phi::errors::InvalidArgument(
              "Variable value (index) of OP(index_add) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              input_dim[axis],
              index_data[i]));
    }
  }

  std::string reducer = reduce;
  if (overwrite) {
    reducer = "assign";
  }

  IndexReduceKernel<T, Context>(
      ctx, x, new_index, updates, axis, reducer, include_self, out);
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
