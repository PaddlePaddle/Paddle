// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_fill_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/index_fill_util.h"

namespace phi {

template <typename T>
__global__ void IndexFillCudaKernel(
    const T* x,
    const int64_t* index,      // 1D 索引数组
    const int64_t index_size,  // 索引数量
    const int dim,             // 操作维度
    const int64_t outer_size,  // dim 之前的维度乘积
    const int64_t dim_size,    // dim 维度的大小
    const int64_t inner_size,  // dim 之后的维度乘积
    const T fill_value,
    T* out) {
  // 总共需要处理: index_size * outer_size * inner_size 个元素
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t total = index_size * outer_size * inner_size;
  if (idx >= total) return;

  // 逆向分解索引idx = outer_idx × (index_size × inner_size)
  //            + index_idx × inner_size
  //            + inner_idx
  int64_t inner_idx = idx % inner_size;
  int64_t temp = idx / inner_size;
  int64_t index_idx = temp % index_size;
  int64_t outer_idx = temp / index_size;

  // 获取 dim 维度的实际索引
  int64_t dim_idx = index[index_idx];
  if (dim_idx < 0) dim_idx += dim_size;

  if (dim_idx < 0 || dim_idx >= dim_size) return;
  // 计算输出偏移
  int64_t offset =
      outer_idx * dim_size * inner_size + dim_idx * inner_size + inner_idx;

  out[offset] = fill_value;
}

template <typename T, typename Context>
void LaunchIndexFillCudaKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               int dim,
                               const DenseTensor& index,
                               const Scalar& value,
                               DenseTensor* out) {
  // 1. 获取输入数据指针
  auto* x_data = x.data<T>();
  T fill_value = value.to<T>();

  // 2. 分配输出内存，并复制输入数据
  bool is_initialized = out->initialized();
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (!is_initialized || (x.data<T>() != out->data<T>())) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  // 3. 获取 index 数据
  auto* index_data = index.data<int64_t>();
  int64_t index_size = index.numel();

  // 如果 index 为空，直接返回
  if (index_size == 0) {
    return;
  }

  // 4. 计算张量维度信息
  auto x_dims = x.dims();
  const int rank = x_dims.size();

  // 处理负数 dim
  if (dim < 0) {
    dim += rank;
  }

  // 5. 计算 outer_size, dim_size, inner_size
  int64_t outer_size = 1;
  int64_t inner_size = 1;
  int64_t dim_size = x_dims[dim];

  for (int i = 0; i < dim; ++i) {
    outer_size *= x_dims[i];
  }
  for (int i = dim + 1; i < rank; ++i) {
    inner_size *= x_dims[i];
  }

  // 6. 计算需要处理的元素总数
  int64_t numel = outer_size * index_size * inner_size;

  // 7. 启动 kernel
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  IndexFillCudaKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          x_data,
          index_data,
          index_size,
          dim,
          outer_size,
          dim_size,
          inner_size,
          fill_value,
          out_data);
}

template <typename T, typename Context>
void IndexFillKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     int dim,
                     const Scalar& value,
                     DenseTensor* out) {
  // 1. 处理空输出的情况
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  // 2. 获取维度信息
  auto x_dims = x.dims();
  const int rank = x_dims.size();

  // 3. 处理负数 dim
  int real_dim = dim;
  if (real_dim < 0) {
    real_dim += rank;
  }

  // 4. 参数校验
  PADDLE_ENFORCE_GE(real_dim,
                    0,
                    common::errors::InvalidArgument(
                        "The dim must be >= -%d and < %d, but received %d.",
                        rank,
                        rank,
                        dim));
  PADDLE_ENFORCE_LT(real_dim,
                    rank,
                    common::errors::InvalidArgument(
                        "The dim must be >= -%d and < %d, but received %d.",
                        rank,
                        rank,
                        dim));

  PADDLE_ENFORCE_EQ(index.dims().size(),
                    1,
                    common::errors::InvalidArgument(
                        "The index tensor must be 1-D, but received %d-D.",
                        index.dims().size()));

  // 5. 处理 index 为空的情况，直接复制输入到输出
  if (index.numel() == 0) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  // 6. 处理 index 数据类型，转换为 int64_t
  DenseTensor index_int64;
  const DenseTensor* ptr_index = nullptr;

  if (index.dtype() == phi::DataType::INT32) {
    index_int64.Resize(index.dims());
    dev_ctx.template Alloc<int64_t>(&index_int64);

    int64_t index_numel = index.numel();
    auto config =
        phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, index_numel);

    phi::funcs::CastToInt64Kernel<int32_t><<<config.block_per_grid,
                                             config.thread_per_block,
                                             0,
                                             dev_ctx.stream()>>>(
        index.data<int32_t>(), index_int64.data<int64_t>(), index_numel);

    ptr_index = &index_int64;
  } else if (index.dtype() == phi::DataType::INT64) {
    ptr_index = &index;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        phi::DataTypeToString(index.dtype())));
  }

  // 7. 调用 Launch 函数
  LaunchIndexFillCudaKernel<T, Context>(
      dev_ctx, x, real_dim, *ptr_index, value, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(index_fill,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexFillKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
