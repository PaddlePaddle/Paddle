/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/concat_grad_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void ConcatCooGradKernel(const Context& dev_ctx,
                         const std::vector<const SparseCooTensor*>& x,
                         const SparseCooTensor& out_grad,
                         const Scalar& axis_scalar,
                         std::vector<SparseCooTensor*> x_grad) {
  PADDLE_ENFORCE_NOT_NULL(
      x[0], phi::errors::NotFound("The first input tensor is not initalized."));

  auto axis = axis_scalar.to<int64_t>();
  axis = phi::funcs::ComputeAxis(static_cast<int64_t>(axis),
                                 static_cast<int64_t>(x[0]->dims().size()));
  // get output tensor that the name is not kEmptyVarName
  for (auto& t : x_grad) {
    if (t && t->numel() != 0UL) {
      // 已经处理了indices了,同时对应的value的nummel也已经处理
      // 那么在这里只需要额外的处理对应的value的实际值即可
      EmptyLikeCooKernel<T, Context>(dev_ctx, out_grad, t);

    } else {
      t = nullptr;
    }
  }

  std::vector<DenseTensor*> x_grad_values_data_vec;
  std::vector<const DenseTensor*> ref_inputs;
  for (size_t i = 0; i < x_grad.size(); ++i) {
    x_grad_values_data_vec.push_back(x_grad[i]->mutable_values());
    ref_inputs.push_back(&(x[i]->values()));
  }
  funcs::SplitFunctor<Context, T> split_functor;
  split_functor(
      dev_ctx, out_grad.values(), ref_inputs, axis, &x_grad_values_data_vec);
}

template <typename T, typename Context>
void ConcatCsrGradKernel(const Context& dev_ctx,
                         const std::vector<const SparseCsrTensor*>& x,
                         const SparseCsrTensor& out_grad,
                         const Scalar& axis_scalar,
                         std::vector<SparseCsrTensor*> x_grad) {
  PADDLE_ENFORCE_NOT_NULL(
      x[0], phi::errors::NotFound("The first input tensor is not initalized."));
  const size_t num_split = x.size();
  if (num_split == 1) {
    phi::Copy<Context>(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad[0]);
    return;
  }

  auto axis = axis_scalar.to<int>();
  axis = phi::funcs::ComputeAxis(static_cast<int64_t>(axis),
                                 static_cast<int64_t>(x[0]->dims().size()));
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument("concat_grad: axis should be larger than or "
                                   "equal to 0, but received axis is %d.",
                                   axis));
  for (auto& t : x_grad) {
    if (t && t->numel() != 0UL) {
      // 不需要处理 crows和cols,在empty中会处理这里的复制
      // 另外value的长度也会在这里处理
      EmptyLikeCsrKernel<T, Context>(dev_ctx, out_grad, t);

    } else {
      t = nullptr;
    }
  }

  T* out_values_data = out_grad.values().data<T>();
  // std::vector<const T*> x_values_data_vec;

  // x_values_data_vec.reserve(num_split);
  int x_dim = x[0]->dims().size();

  if (x_dim == 2) {
    if (axis == 0) {
      int64_t value_offset = 0;
      for (size_t i = 0; i != num_split; i++) {
        int64_t nnz = x[i]->nnz();

        DenseTensor* x_grad_value = x_grad[i]->mutable_values();
        // csr的value是只有一维的densetensor
        std::memcpy(x_grad_value->data<T>(),
                    out_values_data + value_offset,
                    nnz * sizeof(T));
        value_offset += nnz;
      }
    } else {
      int64_t crow_numel = x[0]->crows().numel();
      T* now_grad_values_data = nullptr;
      int out_cols_offset = 0, now_cols_offset = 0;
      for (int64_t j = 1; j < crow_numel; j++) {
        for (int64_t i = 0; i != num_split; i++) {
          now_cols_offset = 0;
          auto* crows_data = x[i]->crows().data<int64_t>();

          now_grad_values_data = x_grad[i]->mutable_values()->data<T>();
          for (int64_t k = 0; crows_data[j] - crows_data[j - 1]; k++) {
            now_grad_values_data[now_cols_offset] =
                out_values_data[out_cols_offset];
            now_cols_offset++;
            out_cols_offset++;
          }
        }
      }
    }
  } else {  // dim == 3
    if (axis == 0) {
      for (size_t i = 0; i != num_split; i++) {
        // 把x对应的稀疏值横纵坐标复制到dx上

        phi::Copy<Context>(dev_ctx,
                           x[i]->values(),
                           dev_ctx.GetPlace(),
                           false,
                           x_grad[i]->mutable_values());
      }
    } else if (axis == 1) {
      int64_t batch = static_cast<int>(x[0]->dims()[0]);
      std::vector<int64_t> nnz_vec;
      int64_t value_offset = 0;
      for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i != num_split; i++) {
          int64_t rows = static_cast<int>(x[i]->dims()[1]);
          int64_t* x_crows_ptr = x[i]->crows().data<int64_t>();
          int64_t x_crows_nnz = x_crows_ptr[(b + 1) * (rows + 1) - 1];
          T* x_grad_values_data = x_grad[i]->values().data<int64_t>();
          if (x_crows_nnz) {
            std::memcpy(x_grad_values_data,
                        out_values_data + value_offset,
                        x_crows_nnz * sizeof(T));
          }
          value_offset += x_crows_nnz;
        }
      }
    } else {
      int64_t batch = static_cast<int64_t>(x[0]->dims()[0]);
      int64_t rows = static_cast<int64_t>(x[0]->dims()[1]);
      int64_t now_crow_numel = rows + 1;
      int out_cols_offset = 0;
      int now_cols_offset = 0;
      T* x_grad_values_data = nullptr;
      for (int64_t b = 0; b < batch; b++) {
        now_cols_offset = 0;
        for (int64_t j = 1; j < now_crow_numel; j++) {
          for (int64_t i = 0; i != num_split; i++) {
            x_grad_values_data = x[i]->values().data<int64_t>();
            x_grad_values_data[now_cols_offset] =
                out_values_data[out_cols_offset];
            now_cols_offset++;
            out_cols_offset++;
          }
        }
      }
    }
  }
}
// CSR grad 可以参考element 那一段
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(concat_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCooGradKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t) {}
