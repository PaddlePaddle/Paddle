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

#include "paddle/fluid/operators/gather_scatter_kernel.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

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
__global__ void GatherScatterGPUKernel(tensor_t* self_data,
                                       int dim,
                                       const index_t* index_data,
                                       tensor_t* src_data,
                                       int64_t inner_dim_size,
                                       int select_dim_size,
                                       int replaced_select_dim_size,
                                       int64_t outer_dim_size,
                                       int64_t numel,
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
  int64_t replace_index = k + index * outer_dim_size +
                          i * outer_dim_size * replaced_select_dim_size;
  int64_t self_idx = is_scatter_like ? replace_index : tid;
  int64_t src_idx = is_scatter_like ? tid : replace_index;
  reduce_op(static_cast<tensor_t*>(self_data + self_idx),
            static_cast<tensor_t*>(src_data + src_idx));
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
                  const platform::DeviceContext& ctx) {
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
    int replaced_select_dim_size =
        is_scatter_like ? self_dims[dim] : src_dims[dim];
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }

    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
    }

    int block = 512;
    int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
    int64_t grid = (n + block - 1) / block;
    auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
    GatherScatterGPUKernel<tensor_t, index_t, func_t, is_scatter_like>
        <<<grid, block, 0, stream>>>(self_data,
                                     dim,
                                     index_data,
                                     src_data,
                                     inner_dim_size,
                                     select_dim_size,
                                     replaced_select_dim_size,
                                     outer_dim_size,
                                     index_size,
                                     reduce_op);
  }
};  // struct gpu_gather_scatter_functor

template <typename tensor_t, typename index_t>
void gpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       const platform::DeviceContext& ctx) {
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
                               const platform::DeviceContext& ctx) {
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
                            const platform::DeviceContext& ctx) {
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
                            const platform::DeviceContext& ctx) {
  gpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mul_gpu", reduce_mul, ctx);
}

template <typename tensor_t, typename index_t>
__global__ void ScatterInputGradGPUKernel(tensor_t* grad_data,
                                          int dim,
                                          const index_t* index_data,
                                          int64_t inner_dim_size,
                                          int select_dim_size,
                                          int grad_select_dim_size,
                                          int64_t outer_dim_size,
                                          int64_t numel) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j, k;
  i = tid / (select_dim_size * outer_dim_size);
  int64_t remind = tid % (select_dim_size * outer_dim_size);
  j = remind / outer_dim_size;
  k = remind % outer_dim_size;
  index_t index = index_data[tid];
  int64_t replace_index =
      k + index * outer_dim_size + i * outer_dim_size * grad_select_dim_size;
  grad_data[replace_index] = 0;
}
template <typename tensor_t, typename index_t>
void gpu_scatter_input_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   const platform::DeviceContext& ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto grad_dims = grad.dims();
  int64_t index_size = index.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int select_dim_size = index_dims[dim];
  int grad_select_dim_size = grad_dims[dim];
  for (int64_t i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
  }

  int block = 512;
  int64_t n = inner_dim_size * select_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();

  ScatterInputGradGPUKernel<tensor_t, index_t>
      <<<grid, block, 0, stream>>>(grad_data,
                                   dim,
                                   index_data,
                                   inner_dim_size,
                                   select_dim_size,
                                   grad_select_dim_size,
                                   outer_dim_size,
                                   index_size);
}
Instantiate_Template_Function(gpu_gather_kernel)
    Instantiate_Template_Function(gpu_scatter_assign_kernel)
        Instantiate_Template_Function(gpu_scatter_add_kernel)
            Instantiate_Template_Function(gpu_scatter_mul_kernel)
                Instantiate_Template_Function(gpu_scatter_input_grad_kernel)

}  // namespace operators
}  // namespace paddle
