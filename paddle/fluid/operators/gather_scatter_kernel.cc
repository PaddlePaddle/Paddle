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

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class TensorAssign {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    // VLOG(3) <<">>>TensorAssign>>>";//<< *self_data;
    *self_data = *src_data;
    VLOG(3) << "self_data assigned:" << *self_data;
  }
};
static TensorAssign tensor_assign;

class ReduceAdd {
 public:
  template <typename tensor_t>
  constexpr void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data += *src_data;
  }
};

static ReduceAdd reduce_add;

class ReduceMultiply {
 public:
  template <typename tensor_t>
  constexpr void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data *= *src_data;
  }

  constexpr void operator()(bool* self_data, bool* src_data) const {
    *self_data = *self_data && *src_data;
  }
};

template <typename tensor_t, typename index_t = int64_t,
          bool is_scatter_like = true>
struct cpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(Tensor self, int dim, const Tensor& index, const Tensor& src,
                  const std::string& method_name, const func_t& kernel_func) {
    if (index.numel() == 0) {
      return;
    }
    // gather_scatter_dtype_check(method_name, self, index, src);
    // if (is_scatter_like) {
    //   scatter_shape_check(self, dim, index, self);
    // }
    // else {
    //   gather_shape_check(self, dim, index, self);
    // }

    // auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    // auto index_strides = ensure_nonempty_vec(index.strides().vec());

    // index_sizes[dim] = 1;
    // index_strides[dim] = 0;
    VLOG(3) << "111111111";
    auto* self_data = self.data<tensor_t>();  // problem occour here
    // VLOG(3) << "self_data:" << *self_data;
    VLOG(3) << "222222222";
    auto* index_data = index.data<index_t>();
    VLOG(3) << "index_data:" << *index_data;
    auto* src_data = src.data<tensor_t>();
    VLOG(3) << "xxxxx222222";
    int64_t self_size = self.numel();
    int64_t index_size = index.numel();
    int64_t src_size = src.numel();
    // auto self_dims = self.dims();
    auto index_dims = index.dims();
    // auto src_dims = src.dims();
    VLOG(3) << "3333333";
    if (self_size == 0 || src_size == 0 || index_size == 0) return;
    int select_dim_size = index_dims[dim];
    int64_t inner_dim_size = 1;
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
      inner_dim_size *= index_dims[i];
    }
    VLOG(3) << "444444";

    for (int i = dim + 1; i < index_dims.size(); i++) {
      outer_dim_size *= index_dims[i];
    }
    VLOG(3) << "inner_dim_size:" << inner_dim_size;
    VLOG(3) << "select_dim_size:" << select_dim_size;
    VLOG(3) << "outer_dim_size:" << outer_dim_size;
    VLOG(3) << "index_size:" << outer_dim_size;

    int64_t index_idx = 0;
    int64_t self_idx, src_idx;

    // N layer loop squeezed into 3 layers loop
    for (int64_t i = 0; i < inner_dim_size; i++) {
      for (int64_t j = 0; j < select_dim_size; j++) {
        for (int64_t k = 0; k < outer_dim_size; k++) {
          int64_t index = index_data[index_idx];
          VLOG(3) << "55555 index:" << index;
          /*
            gather computation formula:

            out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

            scatter computation formula:

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

          */
          int64_t replace_index =
              k + index * outer_dim_size + i * outer_dim_size * select_dim_size;
          self_idx = is_scatter_like ? replace_index : index_idx;
          src_idx = is_scatter_like ? index_idx : replace_index;

          kernel_func(reinterpret_cast<tensor_t*>(self_data + self_idx),
                      reinterpret_cast<tensor_t*>(src_data + src_idx));
          index_idx++;
          VLOG(3) << "55555 index_idx:" << index_idx;
        }
      }
    }
  }
};

template <typename tensor_t, typename index_t>
void cpu_gather_kernel(const Tensor& input, int dim, const Tensor& index,
                       Tensor result) {
  VLOG(3) << "input data:" << *(input.data<tensor_t>());
  VLOG(3) << "result data:" << *(result.data<tensor_t>());
  cpu_gather_scatter_functor<tensor_t, index_t,
                             /*is_scatter_like=*/false>()(
      result, dim, index, input, "gather_out_cpu", tensor_assign);
  VLOG(3) << "<<<< Done cpu_gather_kernel <<<<<";
}

template <typename tensor_t, typename index_t>
void cpu_scatter_add_kernel(const Tensor& input, int dim, const Tensor& index,
                            Tensor result) {
  VLOG(3) << "input data:" << *(input.data<tensor_t>());
  VLOG(3) << "result data:" << *(result.data<tensor_t>());
  cpu_gather_scatter_functor<tensor_t, index_t,
                             /*is_scatter_like=*/true>()(
      input, dim, index, result, "gather_out_cpu", reduce_add);
  VLOG(3) << "<<<< Done cpu_scatter_add_kernel <<<<<";
}

namespace plat = paddle::platform;

Instantiate_Template_Funtion(cpu_gather_kernel)
    Instantiate_Template_Funtion(cpu_scatter_add_kernel)

}  // namespace operators
}  // namespace paddle
