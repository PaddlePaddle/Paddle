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

#include "glog/logging.h"

#include "paddle/phi/core/macros.h"

namespace phi {
namespace funcs {

class TensorAssign {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

class ReduceAdd {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data += *src_data;
  }
};
static ReduceAdd reduce_add;

class ReduceMultiply {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data *= *src_data;
  }
};
static ReduceMultiply reduce_mul;

template <typename tensor_t,
          typename index_t = int64_t,
          bool is_scatter_like = true>
struct cpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(phi::DenseTensor self,
                  int dim,
                  const phi::DenseTensor& index UNUSED,
                  const phi::DenseTensor& src,
                  const std::string& method_name UNUSED,
                  const func_t& reduce_op,
                  const phi::DeviceContext& ctx UNUSED) {
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
    if (self_size == 0 || src_size == 0 || index_size == 0) {
      VLOG(3) << "zero size input found";
      phi::errors::InvalidArgument(
          "self_size, src_size, index_size cannot be 0");
      return;
    }
    int64_t select_dim_size = index_dims[dim];
    // index matrix has different shape with self matrix or src matrix.
    int self_select_dim_size = self_dims[dim];
    int src_select_dim_size = src_dims[dim];
    int64_t outer_dim_size_self = 1;
    int64_t outer_dim_size_src = 1;
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }

    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
      outer_dim_size_self *= self_dims[i];
      outer_dim_size_src *= src_dims[i];
    }
    int64_t index_idx = 0;
    // N layer loop squeezed into 3 layers loop
    for (int64_t i = 0; i < inner_dim_size; i++) {
      for (int64_t j = 0; j < select_dim_size; j++) {
        for (int64_t k = 0; k < outer_dim_size; k++) {
          int64_t index = index_data[index_idx];

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

          // This index might out of bound of index matrix's index, so here
          // multiply the replaced_select_dim_size.
          int64_t replace_index_self, replace_index_src;
          if (is_scatter_like) {
            replace_index_self = k + index * outer_dim_size_self +
                                 i * outer_dim_size_self * self_select_dim_size;

            replace_index_src = k + j * outer_dim_size_src +
                                i * outer_dim_size_src * src_select_dim_size;
          } else {
            replace_index_self = index_idx;

            replace_index_src = k + index * outer_dim_size_src +
                                i * outer_dim_size_src * src_select_dim_size;
          }
          reduce_op((tensor_t*)(self_data + replace_index_self),  // NOLINT
                    (tensor_t*)(src_data + replace_index_src));   // NOLINT
          index_idx++;
        }
      }
    }
  }
};

template <typename tensor_t, typename index_t>
void cpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/false>()(
      result, dim, index, self, "gather_out_cpu", tensor_assign, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_assign_cpu", tensor_assign, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_add_cpu", reduce_add, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mul_cpu", reduce_mul, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_input_grad_kernel(phi::DenseTensor self UNUSED,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor output,
                                   const phi::DeviceContext& ctx UNUSED) {
  auto* index_data = index.data<index_t>();
  auto* output_data = output.data<tensor_t>();

  auto index_dims = index.dims();
  auto output_dims = output.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_data = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t output_select_dim_size = output_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_data *= output_dims[i];
  }

  int64_t index_idx = 0;
  for (int64_t i = 0; i < inner_dim_size; i++) {
    for (int64_t j = 0; j < select_dim_size; j++) {
      for (int64_t k = 0; k < outer_dim_size; k++) {
        int64_t index = index_data[index_idx];
        int64_t replace_index =
            k + index * outer_dim_size_data +
            i * outer_dim_size_data * output_select_dim_size;
        output_data[replace_index] = 0;
        index_idx++;
      }
    }
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_value_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor output,
                                   const phi::DeviceContext& ctx UNUSED) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* output_data = output.data<tensor_t>();

  auto index_dims = index.dims();
  auto self_dims = self.dims();
  auto output_dims = output.dims();

  int64_t self_size = self.numel();
  bool* is_self_grad_used = new bool[self_size];

  for (int i = 0; i < self_size; i++) {
    is_self_grad_used[i] = false;
  }

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_self = 1;
  int64_t outer_dim_size_output = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t self_select_dim_size = self_dims[dim];
  int64_t output_select_dim_size = output_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_self *= self_dims[i];
    outer_dim_size_output *= output_dims[i];
  }

  int64_t index_idx = index.numel() - 1;
  for (int64_t i = inner_dim_size - 1; i >= 0; i--) {
    for (int64_t j = select_dim_size - 1; j >= 0; j--) {
      for (int64_t k = outer_dim_size - 1; k >= 0; k--) {
        int64_t index = index_data[index_idx];
        int64_t replace_index_self =
            k + index * outer_dim_size_self +
            i * outer_dim_size_self * self_select_dim_size;
        int64_t replace_index_output =
            k + j * outer_dim_size_output +
            i * outer_dim_size_output * output_select_dim_size;
        if (!is_self_grad_used[replace_index_self]) {
          output_data[replace_index_output] = self_data[replace_index_self];
          is_self_grad_used[replace_index_self] = true;
        } else {
          output_data[replace_index_output] = 0;
        }
        index_idx--;
      }
    }
  }
  delete[] is_self_grad_used;
}

Instantiate_Template_Function(cpu_gather_kernel)
    Instantiate_Template_Function(cpu_scatter_assign_kernel)
        Instantiate_Template_Function(cpu_scatter_add_kernel)
            Instantiate_Template_Function(cpu_scatter_mul_kernel)
                Instantiate_Template_Function(cpu_scatter_input_grad_kernel)
                    Instantiate_Template_Function(cpu_scatter_value_grad_kernel)

}  // namespace funcs
}  // namespace phi
