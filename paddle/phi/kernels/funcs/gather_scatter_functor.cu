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

#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace phi {
namespace funcs {

class TensorAssign {
 public:
  template <typename tensor_t>
  constexpr void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

class ReduceAdd {
 public:
  template <
      typename tensor_t,
      std::enable_if_t<!std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    phi::CudaAtomicAdd(self_data, *src_data);
  }
  template <typename tensor_t,
            std::enable_if_t<std::is_same<tensor_t, uint8_t>::value>* = nullptr>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data += *src_data;
  }
};
static ReduceAdd reduce_add;

class ReduceMul {
 public:
  template <typename tensor_t>
  __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data *= *src_data;
    // TODO(huangxu96) platform::CudaAtomicMul(*self_data, *src_data);
  }
};
static ReduceMul reduce_mul;

template <typename tensor_t,
          typename index_t,
          typename func_t,
          bool is_scatter_like = true>
__global__ void ScatterAssignGPUKernel(tensor_t* self_data,
                                       int dim,
                                       const index_t* index_data,
                                       tensor_t* src_data,
                                       int select_dim_size,
                                       int self_select_dim_size,
                                       int src_select_dim_size,
                                       int64_t outer_dim_size,
                                       int64_t outer_dim_size_self,
                                       int64_t outer_dim_size_src,
                                       int64_t numel,
                                       int64_t numel_data,
                                       const func_t& reduce_op) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  extern __shared__ int thread_ids[];

  if (tid == 0) {
    for (int i = 0; i < numel_data; i++) {
      thread_ids[i] = 0;
    }
  }
  __syncthreads();
  int64_t i, j, k;  // The i, j, k here is the index of the 3 layers loop
                    // squeezed from the N layers loop.
  /* tid = i * select_dim_size * outer_dim_size + j * outer_dim_size + k */
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  /*
    gather computation formula:

    self[i][j][k] = src[index[i][j][k]][j][k]  # if dim == 0
    self[i][j][k] = src[i][index[i][j][k]][k]  # if dim == 1
    self[i][j][k] = src[i][j][index[i][j][k]]  # if dim == 2

    scatter computation formula:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

  */
  // index matrix has different shape with self matrix or src matrix.
  int64_t replace_index_self, replace_index_src;
  if (is_scatter_like) {
    replace_index_self = k + index * outer_dim_size_self +
                         i * outer_dim_size_self * self_select_dim_size;

    replace_index_src = k + j * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  } else {
    replace_index_self = tid;

    replace_index_src = k + index * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  }

  atomicMax(thread_ids + replace_index_self, tid);
  __syncthreads();

  if (tid == thread_ids[replace_index_self]) {
    reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
              static_cast<tensor_t*>(src_data + replace_index_src));
  }
}

template <typename tensor_t,
          typename index_t,
          typename func_t,
          bool is_scatter_like = true>
__global__ void GatherScatterGPUKernel(tensor_t* self_data,
                                       int dim,
                                       const index_t* index_data,
                                       tensor_t* src_data,
                                       int select_dim_size,
                                       int self_select_dim_size,
                                       int src_select_dim_size,
                                       int64_t outer_dim_size,
                                       int64_t outer_dim_size_self,
                                       int64_t outer_dim_size_src,
                                       int64_t numel,
                                       int64_t numel_data,
                                       const func_t& reduce_op) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;  // The i, j, k here is the index of the 3 layers loop
                    // squeezed from the N layers loop.
  /* tid = i * select_dim_size * outer_dim_size + j * outer_dim_size + k */
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  /*
    gather computation formula:

    self[i][j][k] = src[index[i][j][k]][j][k]  # if dim == 0
    self[i][j][k] = src[i][index[i][j][k]][k]  # if dim == 1
    self[i][j][k] = src[i][j][index[i][j][k]]  # if dim == 2

    scatter computation formula:

    self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

  */
  // index matrix has different shape with self matrix or src matrix.
  int64_t replace_index_self, replace_index_src;
  if (is_scatter_like) {
    replace_index_self = k + index * outer_dim_size_self +
                         i * outer_dim_size_self * self_select_dim_size;

    replace_index_src = k + j * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  } else {
    replace_index_self = tid;

    replace_index_src = k + index * outer_dim_size_src +
                        i * outer_dim_size_src * src_select_dim_size;
  }

  reduce_op(static_cast<tensor_t*>(self_data + replace_index_self),
            static_cast<tensor_t*>(src_data + replace_index_src));
}

template <typename tensor_t,
          typename index_t = int64_t,
          bool is_scatter_like = true>
struct gpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(phi::DenseTensor self,
                  int dim,
                  const phi::DenseTensor& index,
                  phi::DenseTensor src,
                  const std::string& method_name,
                  const func_t& reduce_op,
                  const phi::DeviceContext& ctx) {
    if (index.numel() == 0) {
      return;
    }
    auto* self_data = self.data<tensor_t>();
    auto* index_data = index.data<index_t>();
    auto* src_data = src.data<tensor_t>();
    int64_t self_size = self.numel();
    int64_t index_size = index.numel();
    int64_t src_size = src.numel();
    auto self_dims = self.dims();
    auto index_dims = index.dims();
    auto src_dims = src.dims();
    if (self_size == 0 || src_size == 0 || index_size == 0) return;
    int select_dim_size = index_dims[dim];
    // index matrix has different shape with self matrix or src matrix.
    int self_select_dim_size = self_dims[dim];
    int src_select_dim_size = src_dims[dim];
    int64_t outer_dim_size_self = 1;
    int64_t outer_dim_size_src = 1;
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }

    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
      outer_dim_size_self *= self_dims[i];
      outer_dim_size_src *= src_dims[i];
    }

    int block = 512;
    int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
    int64_t grid = (n + block - 1) / block;
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    if (method_name == "scatter_assign_gpu") {
      int shared_mem_size =
          is_scatter_like ? sizeof(int) * self_size : sizeof(int) * index_size;
      ScatterAssignGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
          <<<grid, block, shared_mem_size, stream>>>(self_data,
                                                     dim,
                                                     index_data,
                                                     src_data,
                                                     select_dim_size,
                                                     self_select_dim_size,
                                                     src_select_dim_size,
                                                     outer_dim_size,
                                                     outer_dim_size_self,
                                                     outer_dim_size_src,
                                                     index_size,
                                                     self_size,
                                                     reduce_op);
    } else {
      GatherScatterGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
          <<<grid, block, 0, stream>>>(self_data,
                                       dim,
                                       index_data,
                                       src_data,
                                       select_dim_size,
                                       self_select_dim_size,
                                       src_select_dim_size,
                                       outer_dim_size,
                                       outer_dim_size_self,
                                       outer_dim_size_src,
                                       index_size,
                                       self_size,
                                       reduce_op);
    }
  }
};  // struct gpu_gather_scatter_functor

template <typename tensor_t, typename index_t>
void gpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/false>()(
      result, dim, index, self, "gather_out_gpu", tensor_assign, ctx);
  return;
}

template <typename tensor_t, typename index_t>
void gpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_assign_gpu", tensor_assign, ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_add_gpu", reduce_add, ctx);
}

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            const phi::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mul_gpu", reduce_mul, ctx);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterInputGradGPUKernel(tensor_t* grad_data,
                                          int dim,
                                          const index_t* index_data,
                                          int select_dim_size,
                                          int grad_select_dim_size,
                                          int64_t outer_dim_size,
                                          int64_t outer_dim_size_data,
                                          int64_t numel,
                                          int64_t numel_data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index = k + index * outer_dim_size_data +
                          i * outer_dim_size_data * grad_select_dim_size;

  grad_data[replace_index] = 0;
}
template <typename tensor_t, typename index_t>
void gpu_scatter_input_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   const phi::DeviceContext& ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto grad_dims = grad.dims();
  int64_t index_size = index.numel();
  int64_t grad_size = grad.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_data = 1;
  int select_dim_size = index_dims[dim];
  int grad_select_dim_size = grad_dims[dim];
  for (int64_t i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_data *= grad_dims[i];
  }

  int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  int shared_mem_size = sizeof(int) * grad_size;
  ScatterInputGradGPUKernel<tensor_t, index_t>
      <<<grid, block, shared_mem_size, stream>>>(grad_data,
                                                 dim,
                                                 index_data,
                                                 select_dim_size,
                                                 grad_select_dim_size,
                                                 outer_dim_size,
                                                 outer_dim_size_data,
                                                 index_size,
                                                 grad_size);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterValueGradGPUKernel(tensor_t* grad_data,
                                          int dim,
                                          const tensor_t* self_data,
                                          const index_t* index_data,
                                          int select_dim_size,
                                          int self_select_dim_size,
                                          int grad_select_dim_size,
                                          int64_t outer_dim_size,
                                          int64_t outer_dim_size_self,
                                          int64_t outer_dim_size_grad,
                                          int64_t numel,
                                          int64_t numel_data) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  extern __shared__ int thread_ids[];

  if (tid == 0) {
    for (int i = 0; i < numel_data; i++) {
      thread_ids[i] = 0;
    }
  }
  __syncthreads();
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index_self = k + index * outer_dim_size_self +
                               i * outer_dim_size_self * self_select_dim_size;

  atomicMax(thread_ids + replace_index_self, tid);
  __syncthreads();

  if (tid == thread_ids[replace_index_self]) {
    int64_t replace_index_grad = k + j * outer_dim_size_grad +
                                 i * outer_dim_size_grad * grad_select_dim_size;
    grad_data[replace_index_grad] = self_data[replace_index_self];
  }
}
template <typename tensor_t, typename index_t>
void gpu_scatter_value_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   const phi::DeviceContext& ctx) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto self_dims = self.dims();
  auto grad_dims = grad.dims();
  int64_t index_size = index.numel();
  int64_t self_size = self.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_self = 1;
  int64_t outer_dim_size_grad = 1;
  int select_dim_size = index_dims[dim];
  int self_select_dim_size = self_dims[dim];
  int grad_select_dim_size = grad_dims[dim];
  for (int64_t i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_self *= self_dims[i];
    outer_dim_size_grad *= grad_dims[i];
  }

  int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  int shared_mem_size = sizeof(int) * self_size;
  ScatterValueGradGPUKernel<tensor_t, index_t>
      <<<grid, block, shared_mem_size, stream>>>(grad_data,
                                                 dim,
                                                 self_data,
                                                 index_data,
                                                 select_dim_size,
                                                 self_select_dim_size,
                                                 grad_select_dim_size,
                                                 outer_dim_size,
                                                 outer_dim_size_self,
                                                 outer_dim_size_grad,
                                                 index_size,
                                                 self_size);
}
Instantiate_Template_Function(gpu_gather_kernel)
    Instantiate_Template_Function(gpu_scatter_assign_kernel)
        Instantiate_Template_Function(gpu_scatter_add_kernel)
            Instantiate_Template_Function(gpu_scatter_mul_kernel)
                Instantiate_Template_Function(gpu_scatter_input_grad_kernel)
                    Instantiate_Template_Function(gpu_scatter_value_grad_kernel)
}  // namespace funcs
}  // namespace phi
