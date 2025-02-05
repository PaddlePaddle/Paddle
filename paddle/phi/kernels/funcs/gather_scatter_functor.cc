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

#include "paddle/common/macros.h"

namespace phi::funcs {

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

class ReduceMax {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data > *self_data ? *src_data : *self_data;
  }
};
static ReduceMax reduce_max;

class ReduceMin {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data < *self_data ? *src_data : *self_data;
  }
};
static ReduceMin reduce_min;

template <typename tensor_t,
          typename index_t = int64_t,
          bool is_scatter_like = true>
struct cpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(phi::DenseTensor self,
                  int dim,
                  const phi::DenseTensor& index,
                  const phi::DenseTensor& src,
                  const std::string& method_name,
                  const func_t& reduce_op,
                  bool include_self,
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
      common::errors::InvalidArgument(
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
    std::vector<int> nums_of_elements(self.numel(), 0);
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
            // scatter
            PADDLE_ENFORCE_GE(
                index,
                -self_select_dim_size,
                common::errors::OutOfRange(
                    "Variable value (index) of OP(take_along_axis) "
                    "expected >= %d and < %d, but got %ld."
                    "Please check the input "
                    "value.",
                    -self_select_dim_size,
                    self_select_dim_size,
                    index));
            PADDLE_ENFORCE_LT(
                index,
                self_select_dim_size,
                common::errors::OutOfRange(
                    "Variable value (index) of OP(take_along_axis) "
                    "expected >= %d and < %d, but got %ld."
                    "Please check the input "
                    "value.",
                    -self_select_dim_size,
                    self_select_dim_size,
                    index));
            if (index < 0) {
              index += self_select_dim_size;
            }
            replace_index_self = k + index * outer_dim_size_self +
                                 i * outer_dim_size_self * self_select_dim_size;

            replace_index_src = k + j * outer_dim_size_src +
                                i * outer_dim_size_src * src_select_dim_size;
          } else {
            // gather
            PADDLE_ENFORCE_GE(
                index,
                -src_select_dim_size,
                common::errors::OutOfRange(
                    "Variable value (index) of OP(take_along_axis) "
                    "expected >= %ld and < %ld, but got %ld. "
                    "Please check the input "
                    "value.",
                    -src_select_dim_size,
                    src_select_dim_size,
                    index));
            PADDLE_ENFORCE_LT(
                index,
                src_select_dim_size,
                common::errors::OutOfRange(
                    "Variable value (index) of OP(take_along_axis) "
                    "expected >= %ld and < %ld, but got %ld. "
                    "Please check the input "
                    "value.",
                    -src_select_dim_size,
                    src_select_dim_size,
                    index));
            if (index < 0) {
              index += src_select_dim_size;
            }
            replace_index_self = index_idx;

            replace_index_src = k + index * outer_dim_size_src +
                                i * outer_dim_size_src * src_select_dim_size;
          }
          if (include_self == false &&
              nums_of_elements[replace_index_self] == 0) {
            self_data[replace_index_self] = src_data[replace_index_src];
          } else {
            reduce_op((tensor_t*)(self_data + replace_index_self),  // NOLINT
                      (tensor_t*)(src_data + replace_index_src));   // NOLINT
          }
          nums_of_elements[replace_index_self] += 1;
          index_idx++;
        }
      }
    }
    if (method_name == "scatter_mean_cpu") {
      for (int i = 0; i < self_size; i++) {
        if (nums_of_elements[i]) {
          if (include_self) {
            self_data[i] =
                self_data[i] / static_cast<tensor_t>(nums_of_elements[i] + 1);
          } else {
            self_data[i] =
                self_data[i] / static_cast<tensor_t>(nums_of_elements[i]);
          }
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
                       bool include_self,
                       const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/false>()(result,
                                                          dim,
                                                          index,
                                                          self,
                                                          "gather_out_cpu",
                                                          tensor_assign,
                                                          include_self,
                                                          ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               bool include_self,
                               const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(self,
                                                         dim,
                                                         index,
                                                         src,
                                                         "scatter_assign_cpu",
                                                         tensor_assign,
                                                         include_self,
                                                         ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_add_cpu", reduce_add, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mul_cpu", reduce_mul, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_kernel(phi::DenseTensor self,
                             int dim,
                             const phi::DenseTensor& index,
                             phi::DenseTensor src,
                             bool include_self,
                             const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_mean_cpu", reduce_add, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_max_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_max_cpu", reduce_max, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_min_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "scatter_min_cpu", reduce_min, include_self, ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_input_grad_kernel(phi::DenseTensor self UNUSED,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self UNUSED,
                                   const phi::DeviceContext& ctx UNUSED) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto grad_dims = grad.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_data = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_data *= grad_dims[i];
  }

  int64_t index_idx = 0;
  for (int64_t i = 0; i < inner_dim_size; i++) {
    for (int64_t j = 0; j < select_dim_size; j++) {
      for (int64_t k = 0; k < outer_dim_size; k++) {
        int64_t index = index_data[index_idx];
        int64_t replace_index = k + index * outer_dim_size_data +
                                i * outer_dim_size_data * grad_select_dim_size;
        grad_data[replace_index] = 0;
        index_idx++;
      }
    }
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_min_max_input_grad_kernel(phi::DenseTensor self UNUSED,
                                               int dim,
                                               const phi::DenseTensor& index,
                                               const phi::DenseTensor& out,
                                               const phi::DenseTensor& x,
                                               const phi::DenseTensor& value,
                                               phi::DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self UNUSED,
                                               const phi::DeviceContext& ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();
  auto* out_data = out.data<tensor_t>();
  auto* x_data = x.data<tensor_t>();
  auto* value_data = value.data<tensor_t>();

  int64_t grad_size = grad.numel();
  auto index_dims = index.dims();
  auto grad_dims = grad.dims();
  auto value_dims = value.dims();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_grad = 1;
  int64_t outer_dim_size_value = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  int64_t value_select_dim_size = value_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_grad *= grad_dims[i];
    outer_dim_size_value *= value_dims[i];
  }

  int64_t index_idx = 0;
  std::vector<int> num_elements(grad_size, 0);
  for (int64_t i = 0; i < inner_dim_size; i++) {
    for (int64_t j = 0; j < select_dim_size; j++) {
      for (int64_t k = 0; k < outer_dim_size; k++) {
        int64_t index = index_data[index_idx];
        int64_t replace_index_grad =
            k + index * outer_dim_size_grad +
            i * outer_dim_size_grad * grad_select_dim_size;
        if ((reduce == "multiply" || reduce == "mul") &&
            num_elements[replace_index_grad] == 0) {
          grad_data[replace_index_grad] = static_cast<tensor_t>(
              grad_data[replace_index_grad] * out_data[replace_index_grad] /
              x_data[replace_index_grad]);
          num_elements[replace_index_grad] += 1;
        } else if (reduce == "amin" || reduce == "amax") {
          if (out_data[replace_index_grad] != x_data[replace_index_grad]) {
            grad_data[replace_index_grad] = 0;
          } else {
            int64_t replace_index_value =
                k + j * outer_dim_size_value +
                i * outer_dim_size_value * value_select_dim_size;
            if (out_data[replace_index_grad] == value_data[replace_index_value])
              num_elements[replace_index_grad] += 1;
          }
        }
        index_idx++;
      }
    }
  }
  if (reduce == "amin" || reduce == "amax") {
    for (int64_t i = 0; i < grad_size; i++) {
      grad_data[i] = grad_data[i] / static_cast<tensor_t>(num_elements[i] + 1);
    }
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_input_grad_kernel(phi::DenseTensor self UNUSED,
                                        int dim,
                                        const phi::DenseTensor& index,
                                        phi::DenseTensor grad,
                                        bool include_self UNUSED,
                                        const phi::DeviceContext& ctx UNUSED) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto grad_dims = grad.dims();

  int64_t grad_size = grad.numel();

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_data = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_data *= grad_dims[i];
  }

  int64_t index_idx = 0;
  std::vector<int> num_elements(grad_size, 0);
  for (int64_t i = 0; i < inner_dim_size; i++) {
    for (int64_t j = 0; j < select_dim_size; j++) {
      for (int64_t k = 0; k < outer_dim_size; k++) {
        int64_t index = index_data[index_idx];
        int64_t replace_index = k + index * outer_dim_size_data +
                                i * outer_dim_size_data * grad_select_dim_size;
        num_elements[replace_index] += 1;
        index_idx++;
      }
    }
  }
  for (int64_t i = 0; i < grad_size; i++)
    if (num_elements[i])
      grad_data[i] = grad_data[i] / static_cast<tensor_t>(num_elements[i] + 1);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_value_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self UNUSED,
                                   const phi::DeviceContext& ctx UNUSED) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto self_dims = self.dims();
  auto grad_dims = grad.dims();

  int64_t self_size = self.numel();
  std::vector<bool> is_self_grad_used(self_size, false);

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_self = 1;
  int64_t outer_dim_size_grad = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t self_select_dim_size = self_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_self *= self_dims[i];
    outer_dim_size_grad *= grad_dims[i];
  }
  int64_t index_idx = index.numel() - 1;
  for (int64_t i = inner_dim_size - 1; i >= 0; i--) {
    for (int64_t j = select_dim_size - 1; j >= 0; j--) {
      for (int64_t k = outer_dim_size - 1; k >= 0; k--) {
        int64_t index = index_data[index_idx];
        int64_t replace_index_self =
            k + index * outer_dim_size_self +
            i * outer_dim_size_self * self_select_dim_size;
        int64_t replace_index_grad =
            k + j * outer_dim_size_grad +
            i * outer_dim_size_grad * grad_select_dim_size;
        if (!is_self_grad_used[replace_index_self]) {
          grad_data[replace_index_grad] = self_data[replace_index_self];
          is_self_grad_used[replace_index_self] = true;
        }
        index_idx--;
      }
    }
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_add_mean_value_grad_kernel(
    phi::DenseTensor self,
    int dim,
    const phi::DenseTensor& index,
    const phi::DenseTensor& out UNUSED,
    const phi::DenseTensor& x UNUSED,
    const phi::DenseTensor& value UNUSED,
    phi::DenseTensor grad,
    const std::string& reduce,
    bool include_self,
    const phi::DeviceContext& ctx UNUSED) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  auto index_dims = index.dims();
  auto self_dims = self.dims();
  auto grad_dims = grad.dims();

  int64_t self_size = self.numel();
  int64_t grad_size = grad.numel();
  std::vector<int> num_elements;
  if (reduce == "mean") {
    for (int i = 0; i < self_size; i++) {
      if (include_self)
        num_elements.push_back(1);
      else
        num_elements.push_back(0);
    }
  }

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_self = 1;
  int64_t outer_dim_size_grad = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t self_select_dim_size = self_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_self *= self_dims[i];
    outer_dim_size_grad *= grad_dims[i];
  }
  for (int i = 0; i < grad_size; i++) {
    grad_data[i] = static_cast<tensor_t>(0);
  }
  int64_t index_idx = index.numel() - 1;
  if (reduce == "mean") {
    for (int64_t i = inner_dim_size - 1; i >= 0; i--) {
      for (int64_t j = select_dim_size - 1; j >= 0; j--) {
        for (int64_t k = outer_dim_size - 1; k >= 0; k--) {
          int64_t index = index_data[index_idx];
          int64_t replace_index_self =
              k + index * outer_dim_size_self +
              i * outer_dim_size_self * self_select_dim_size;
          num_elements[replace_index_self] += 1;
          index_idx--;
        }
      }
    }
    index_idx = index.numel() - 1;
  }
  for (int64_t i = inner_dim_size - 1; i >= 0; i--) {
    for (int64_t j = select_dim_size - 1; j >= 0; j--) {
      for (int64_t k = outer_dim_size - 1; k >= 0; k--) {
        int64_t index = index_data[index_idx];
        int64_t replace_index_self =
            k + index * outer_dim_size_self +
            i * outer_dim_size_self * self_select_dim_size;
        int64_t replace_index_grad =
            k + j * outer_dim_size_grad +
            i * outer_dim_size_grad * grad_select_dim_size;
        if (reduce == "add")
          grad_data[replace_index_grad] = self_data[replace_index_self];
        else if (reduce == "mean")
          grad_data[replace_index_grad] =
              self_data[replace_index_self] /
              static_cast<tensor_t>(num_elements[replace_index_self]);
        index_idx--;
      }
    }
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_min_max_value_grad_kernel(phi::DenseTensor self,
                                               int dim,
                                               const phi::DenseTensor& index,
                                               const phi::DenseTensor& out,
                                               const phi::DenseTensor& x,
                                               const phi::DenseTensor& value,
                                               phi::DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const phi::DeviceContext& ctx) {
  auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();
  auto* out_data = out.data<tensor_t>();
  auto* x_data = x.data<tensor_t>();
  auto* value_data = value.data<tensor_t>();

  auto index_dims = index.dims();
  auto self_dims = self.dims();
  auto grad_dims = grad.dims();

  int64_t self_size = self.numel();
  std::vector<int> num_elements;
  if (reduce == "amin" || reduce == "amax") {
    for (int i = 0; i < self_size; i++) {
      num_elements.push_back(0);
    }
  }
  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  int64_t outer_dim_size_self = 1;
  int64_t outer_dim_size_grad = 1;
  int64_t select_dim_size = index_dims[dim];
  int64_t self_select_dim_size = self_dims[dim];
  int64_t grad_select_dim_size = grad_dims[dim];
  for (int i = 0; i < dim; ++i) {
    inner_dim_size *= index_dims[i];
  }

  for (int i = dim + 1; i < index_dims.size(); i++) {
    outer_dim_size *= index_dims[i];
    outer_dim_size_self *= self_dims[i];
    outer_dim_size_grad *= grad_dims[i];
  }
  int64_t index_idx = 0;
  for (int64_t i = 0; i < inner_dim_size; i++) {
    for (int64_t j = 0; j < select_dim_size; j++) {
      for (int64_t k = 0; k < outer_dim_size; k++) {
        int64_t index = index_data[index_idx];
        int64_t replace_index_self =
            k + index * outer_dim_size_self +
            i * outer_dim_size_self * self_select_dim_size;
        int64_t replace_index_grad =
            k + j * outer_dim_size_grad +
            i * outer_dim_size_grad * grad_select_dim_size;
        if ((reduce == "amin" || reduce == "amax") &&
            out_data[replace_index_self] == value_data[replace_index_grad]) {
          num_elements[replace_index_self] += 1;
        } else if (reduce == "mul" || reduce == "multiply") {
          grad_data[replace_index_grad] =
              self_data[replace_index_self] *
              (out_data[replace_index_self] / value_data[replace_index_grad]);
        }
        index_idx++;
      }
    }
  }
  if (reduce == "amin" || reduce == "amax") {
    index_idx = 0;
    for (int64_t i = 0; i < inner_dim_size; i++) {
      for (int64_t j = 0; j < select_dim_size; j++) {
        for (int64_t k = 0; k < outer_dim_size; k++) {
          int64_t index = index_data[index_idx];
          int64_t replace_index_self =
              k + index * outer_dim_size_self +
              i * outer_dim_size_self * self_select_dim_size;
          int64_t replace_index_grad =
              k + j * outer_dim_size_grad +
              i * outer_dim_size_grad * grad_select_dim_size;
          if (out_data[replace_index_self] == value_data[replace_index_grad]) {
            if (out_data[replace_index_self] == x_data[replace_index_self])
              grad_data[replace_index_grad] =
                  self_data[replace_index_self] /
                  static_cast<tensor_t>(num_elements[replace_index_self] + 1);
            else
              grad_data[replace_index_grad] =
                  self_data[replace_index_self] /
                  static_cast<tensor_t>(num_elements[replace_index_self]);
          }
          index_idx++;
        }
      }
    }
  }
}

Instantiate_Template_Function(cpu_gather_kernel)                  // NOLINT
    Instantiate_Template_Function(cpu_scatter_assign_kernel)      // NOLINT
    Instantiate_Template_Function(cpu_scatter_add_kernel)         // NOLINT
    Instantiate_Template_Function(cpu_scatter_mul_kernel)         // NOLINT
    Instantiate_Template_Function(cpu_scatter_mean_kernel)        // NOLINT
    Instantiate_Template_Function(cpu_scatter_max_kernel)         // NOLINT
    Instantiate_Template_Function(cpu_scatter_min_kernel)         // NOLINT
    Instantiate_Template_Function(cpu_scatter_input_grad_kernel)  // NOLINT
    Instantiate_Template_Function(cpu_scatter_value_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        cpu_scatter_mul_min_max_input_grad_kernel)                     // NOLINT
    Instantiate_Template_Function(cpu_scatter_mean_input_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        cpu_scatter_add_mean_value_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        cpu_scatter_mul_min_max_value_grad_kernel)  // NOLINT

}  // namespace phi::funcs
