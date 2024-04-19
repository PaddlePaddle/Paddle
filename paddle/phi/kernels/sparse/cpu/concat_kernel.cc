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

#include "paddle/phi/kernels/sparse/concat_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace phi {
namespace sparse {

static void check_cat_sparse_dims(SparseCooTensor const* t,
                                  int64_t pos,
                                  DDim dims,
                                  int64_t axis,
                                  int64_t sparse_dim,
                                  int64_t dense_dim) {
  PADDLE_ENFORCE_EQ(t->sparse_dim(),
                    sparse_dim,
                    "All tensors must have the same sparse_dim ",
                    sparse_dim,
                    ", but tensor at position ",
                    pos,
                    " has ",
                    t->sparse_dim());
  PADDLE_ENFORCE_EQ(t->dense_dim(),
                    dense_dim,
                    "All tensors must have the same dense_dim ",
                    dense_dim,
                    ", but tensor at position ",
                    pos,
                    " has ",
                    t->dense_dim());
}

template <typename T, typename Context>
void ConcatCooKernel(const Context& dev_ctx,
                     const std::vector<const SparseCooTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCooTensor* out) {
  std::vector<MetaTensor> meta_x;
  meta_x.reserve(x.size());
  std::vector<const MetaTensor*> meta_x_ptr;
  meta_x_ptr.reserve(x.size());
  std::vector<DenseTensor> indices;
  std::vector<DenseTensor> values;
  std::vector<phi::DDim> x_dims;

  int64_t axis = axis_scalar.to<int64_t>();
  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());
  int32_t sparse_dim = x[0]->sparse_dim();
  int32_t dense_dim = x[0]->dense_dim();

  DDim dims = x[0]->dims();
  DenseTensor out_indices;
  DenseTensor out_values;
  funcs::ConcatFunctor<Context, T> concat_functor;
  int64_t pos = 0;
  for (const auto* t : x) {
    check_cat_sparse_dims(t, pos, dims, axis, sparse_dim, dense_dim);
    meta_x_ptr.push_back(&t->meta());
    x_dims.push_back(t->dims());
    pos++;
  }

  MetaTensor meta_out(out);
  ConcatInferMeta(meta_x_ptr, axis, &meta_out);
  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);

  if (axis < sparse_dim) {
    for (const auto* t : x) {
      indices.push_back(t->indices());
      values.push_back(t->values());
    }
    // 因为在前面进行了检查,所以这个维度的nnz都一样
    int64_t out_nnz = x[0]->nnz();
    concat_functor(dev_ctx, indices, static_cast<int>(1), &out_indices);
    concat_functor(dev_ctx, values, static_cast<int>(0), &out_values);
    int64_t col = 0;
    int64_t cumulative_offset = 0;
    auto* out_indices_data = out_indices.data<int64_t>();
    for (size_t i = 0; i != x.size(); i++) {
      int64_t this_piece_size = x[i]->nnz();
      if (i > 0) {
        // 原始的代码
        // out_indices[axis].narrow(0, col, this_piece_size) +=
        // cumulative_offset;
        for (int64_t j = col; j != this_piece_size; j++) {
          out_indices_data[axis * out_nnz + j] += cumulative_offset;
        }
      }
      cumulative_offset += x[i]->dims()[axis];

      col += this_piece_size;
    }
    out->SetMember(out_indices, out_values, out_dims, x[0]->coalesced());
  } else {
    int64_t values_dim = axis - sparse_dim + 1;
    const int64_t total_size =
        std::accumulate(x.begin(),
                        x.end(),
                        static_cast<int64_t>(0),
                        [values_dim](int64_t l, const SparseCooTensor* r) {
                          return l + r->values().dims()[values_dim];
                        });
    DDim zeros_sizes = x[0]->values().dims();
    int64_t cumulative_size = 0;

    for (const auto* t : x) {
      zeros_sizes[0] = t->values().dims()[0];
      zeros_sizes[values_dim] = cumulative_size;
      cumulative_size += t->values().dims()[values_dim];
      // z1 z2是全0的向量
      DenseTensor z1 =
          phi::Full<T, Context>(dev_ctx, common::vectorize(zeros_sizes), 0);
      zeros_sizes[values_dim] = total_size - cumulative_size;
      DenseTensor z2 =
          phi::Full<T, Context>(dev_ctx, common::vectorize(zeros_sizes), 0);
      std::vector<DenseTensor> now_values;
      now_values.push_back(z1);
      now_values.push_back(t->values());
      now_values.push_back(z2);
      auto concat_value =
          std::make_shared<DenseTensor>();  // 创建DenseTensor的智能指针
      concat_functor(dev_ctx, now_values, values_dim, concat_value.get());

      values.push_back(*concat_value);
      indices.push_back(t->indices());
    }
    concat_functor(dev_ctx, indices, static_cast<int>(1), &out_indices);
    concat_functor(dev_ctx, values, static_cast<int>(0), &out_values);

    out->SetMember(out_indices, out_values, out_dims, x[0]->coalesced());
  }
}

template <typename T>
struct csr {
  int64_t crow;
  int64_t col;
  T value;
};

template <typename T, typename Context>
void ConcatCsrKernel(const Context& dev_ctx,
                     const std::vector<const SparseCsrTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCsrTensor* out) {
  const size_t num_split = x.size();
  if (num_split == 1) {
    phi::Copy<Context>(dev_ctx, x[0], dev_ctx.GetPlace(), false, out);
    return;
  }
  std::vector<const MetaTensor*> meta_x_ptr;

  std::vector<phi::DDim> x_dims;
  x_dims.reserve(num_split);

  int64_t axis = axis_scalar.to<int64_t>();
  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());

  DDim dims = x[0]->dims();
  DenseTensor out_cols;
  DenseTensor out_values;
  DenseTensor out_crows;

  int64_t pos = 0;

  std::vector<int64_t> out_value_dims_vec(1, 0);

  std::vector<int64_t> out_crows_dims_vec(1, 0);
  std::vector<int64_t> crows_numel;
  std::vector<int64_t*> crows_data_vec;

  for (const auto* t : x) {
    meta_x_ptr.emplace_back(&t->meta());
    x_dims.emplace_back(t->dims());
    // csr的values,cols crows 是一维数组
    out_value_dims_vec[0] += t->values().numel();
    crows_numel.push_back(t->crows().numel());
    crows_data_vec.push_back(t->crows().data<int64_t>());
  }

  MetaTensor meta_out(out);
  ConcatInferMeta(meta_x_ptr, axis, &meta_out);
  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);
  int x_dim = x_dims[0].size();
  // now csr only support 2-d and 3-d size
  if (x_dim == 2) {
    if (axis == 0) {
      std::vector<DenseTensor> cols;
      cols.reserve(num_split);
      std::vector<DenseTensor> values;
      values.reserve(num_split);
      for (int64_t i = 0; i != num_split; i++) {
        out_crows_dims_vec[0] += crows_numel[i];
        cols.emplace_back(t->cols());
        values.emplace_back(t->values());
      }
      // 减掉额外的0
      out_crows_dims_vec[0] -= x.size() - 1;

      funcs::ConcatFunctor<Context, T> concat_functor;
      out_values->Resize(common::make_ddim(out_value_dims_vec));
      concat_functor(dev_ctx, values, static_cast<int>(0), &out_values);
      // cols的形状与value一致
      out_cols->Resize(common::make_ddim(out_value_dims_vec));
      concat_functor(dev_ctx, cols, static_cast<int>(0), &out_cols);

      out_crows->Resize(common::make_ddim(out_crows_dims_vec));
      auto* out_crows_data = out_crows.data<int64_t>();
      int64_t crow_index = 0;
      // rows_in_slice 保存的是每一个crows的和,方便concat的计算 多给1条方便优化
      std::vector<int64_t> crows_in_slice(num_split + 1, 0);
      // 原始的代码
      // 在cuda中优化?
      for (int64_t i = 0; i != num_split; i++) {
        for (int64_t j = 0; j != crows_numel[i]; j++) {
          crows_in_slice[i + 1] += crows_data_vec[i][j];
          if (j >= 1 || i == 0) {
            // 对于crows.lenght 的序列[[x], [y], [z] ...]
            // 针对输出的情况out_crow应该是[x, y-1, z-1]
            // ,同时y-1,z-1...这些段落要加上之前的叠加和
            // i = 0的时候crows_in_slice[0] = 0
            out_crows_data[j + crow_index] =
                crows_data_vec[i][j] + crows_in_slice[i];
          }
        }

        crow_index += crows_numel[i];
      }
      out->SetMember(out_crows, out_cols, out_values, out_dims);
    } else {  // axis == 1
      out_values->Resize(common::make_ddim(out_value_dims_vec));
      out_cols->Resize(common::make_ddim(out_value_dims_vec));
      // num_split >= 2
      // 先获取最长的crows长度 方便作为基准

      std::vector<T*> values_data_vec;
      std::vector<int64_t*> cols_data_vec;
      int64_t max_crow_numel = 0;
      for (int64_t i = 0; i != num_split; i++) {
        values_data_vec.push_back(values[i].data<T>());
        cols_data_vec.push_back(clos[i].data<int64_t>());
        if (crows_numel[i] > max_crow_numel) {
          max_crow_numel = crows_numel[i];
        }
      }
      if (max_crow_numel == 1) {
        // 为空的情况 为0 的情况 无意义
        phi::Copy<Context>(dev_ctx, x[0], dev_ctx.GetPlace(), false, out);
        return;
      }
      out_crows_dims_vec[0] = max_crow_numel;
      out_crows->Resize(common::make_ddim(out_crows_dims_vec));

      out_cols_data = out_cols.data<int64_t>();
      out_values_data = out_values.data<T>();
      out_crows_data = out_crows.data<int64_t>();

      for (int i = 0; i < max_crow_numel; ++i) {
        out_crows_data[i] = 0;
      }

      int64_t cumulative_cols_offset = 0;
      int now_cols_offset = 0;
      for (int64_t j = 1; j <= max_crow_numel; j++) {
        for (int64_t i = 0; i != num_split; i++) {
          now_cols_offset = 0;
          // 判断条件 貌似可以拿出去
          if (j <= crows_numel[i]) {
            for (int64_t k = 0; crows_data[j] - crows_data[j - 1]; k++) {
              out_crows_data[j]++;
              out_cols_data[cumulative_cols_offset] =
                  values_data_vec[i][now_cols_offset];
              out_values_data[cumulative_cols_offset] =
                  cols_data_vec[i][now_cols_offset];
              now_cols_offset++;
              cumulative_cols_offset++;
            }
          }
        }
      }
      out->SetMember(out_crows, out_cols, out_values, out_dims);
    }

  } else {
    // 参考transpose_kernel.c的方法?? ndim==3的情况
  }
}
}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(concat_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCooKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
