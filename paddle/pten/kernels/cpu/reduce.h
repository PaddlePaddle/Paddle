// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <set>

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/kernels/cast_kernel.h"

#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"
#include "paddle/pten/kernels/funcs/transpose.h"
// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"
namespace pten {

template <typename DeviceContext,
          typename T,
          size_t D,
          size_t R_D,
          typename Functor>
void ReduceFunctor(const DeviceContext& context,
                   const pten::DenseTensor& input,
                   pten::DenseTensor* output,
                   const std::vector<int64_t>& dims,
                   bool keep_dim) {
  auto x = EigenTensor<T, D>::From(input);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto reduce_dim = Eigen::array<int, R_D>();
  std::vector<int64_t> dims_ref = dims;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) dims_ref[i] = x_rank + dims_ref[i];
    reduce_dim[i] = dims_ref[i];
  }
  // construct the squeezed output tensor
  DDim out_dims = output->dims();
  if (keep_dim && x_rank > 1) {
    const int kDelFlag = -2;
    auto dims_vector = paddle::framework::vectorize(out_dims);
    for (size_t i = 0; i < dims_ref.size(); ++i) {
      dims_vector[dims_ref[i]] = kDelFlag;
    }
    dims_vector.erase(remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
                      dims_vector.end());
    out_dims = paddle::framework::make_ddim(dims_vector);
  }
  auto& place = *context.eigen_device();
  Functor functor;

  if (D == 1) {
    auto out = EigenScalar<T>::From(*output);
    functor(place, &x, &out, reduce_dim);
  } else {
    auto out = EigenTensor<T, (D - R_D)>::From(*output, out_dims);
    functor(place, &x, &out, reduce_dim);
  }
}

#define HANDLE_REDUCE_DIM(NDIM, RDIM)                        \
  if (ndim == NDIM && rdim == RDIM) {                        \
    ReduceFunctor<DeviceContext, OutT, NDIM, RDIM, Functor>( \
        dev_ctx, input, output, dims, keep_dim);             \
  }
//////////////// HandleLargeDim

inline void GetShuffledDim(const DDim& src_dims,
                           DDim* dst_dims,
                           const std::vector<int64_t>& reduced_dims,
                           std::vector<int64_t>* perm_axis) {
  // check if it's a reduced dim
  std::vector<bool> src_dims_check(src_dims.size(), false);
  size_t src_size = src_dims.size();
  size_t reduce_size = reduced_dims.size();
  std::vector<int64_t> regular_reduced_dims = reduced_dims;
  for (size_t i = 0; i < regular_reduced_dims.size(); i++) {
    if (regular_reduced_dims[i] < 0) {
      regular_reduced_dims[i] = src_size + regular_reduced_dims[i];
    }
  }

  for (size_t i = 0; i < reduce_size; ++i) {
    dst_dims->at(src_size - reduce_size + i) =
        src_dims[regular_reduced_dims[i]];
    (*perm_axis)[src_size - reduce_size + i] = regular_reduced_dims[i];
    src_dims_check[regular_reduced_dims[i]] = true;
  }

  size_t offset = 0;
  for (size_t i = 0; i < src_dims_check.size(); ++i) {
    bool is_reduced = src_dims_check[i];
    if (!is_reduced) {
      (*perm_axis)[offset] = i;
      dst_dims->at(offset++) = src_dims[i];
    }
  }
}

template <typename DeviceContext, typename OutT>
void GetShuffledInput(const DeviceContext& dev_ctx,
                      const pten::DenseTensor& input,
                      pten::DenseTensor* shuffled_input,
                      const std::vector<int64_t>& dims) {
  DDim shuffled_dims(input.dims());
  std::vector<int64_t> perm_axis(input.dims().size());
  GetShuffledDim(input.dims(), &shuffled_dims, dims, &perm_axis);

  shuffled_input->ResizeAndAllocate(shuffled_dims);
  shuffled_input->mutable_data<OutT>();

  pten::math::TransposeNormal<DeviceContext, OutT> trans;
  trans(dev_ctx, input, shuffled_input, perm_axis);
}

template <typename DeviceContext, typename OutT, typename Functor>
void HandleLargeDim(const DeviceContext& dev_ctx,
                    const pten::DenseTensor& input,
                    pten::DenseTensor* output,
                    const std::vector<int64_t>& dims,
                    bool keep_dim) {
  //  shuffle the reduced dim to the end
  pten::DenseTensor shuffled_input = pten::DenseTensor(
      pten::make_intrusive<paddle::experimental::SharedStorage>(input.place()),
      input.meta());

  GetShuffledInput<DeviceContext, OutT>(dev_ctx, input, &shuffled_input, dims);

  // transpose to 2D tensor whose shape is {unreduced, reduced}.
  const int64_t unreduced = output->numel();
  const int64_t reduced = shuffled_input.numel() / unreduced;
  shuffled_input.ResizeAndAllocate({unreduced, reduced});
  DDim output_dim = output->dims();
  output->ResizeAndAllocate({unreduced});
  ReduceFunctor<DeviceContext, OutT, 2, 1, Functor>(
      dev_ctx, shuffled_input, output, {1}, keep_dim);
  output->ResizeAndAllocate(output_dim);
}

////////////// ReduceKernel

template <typename DeviceContext, typename T, typename OutT, typename Functor>
void ReduceKernelImpl(const DeviceContext& dev_ctx,
                      const pten::DenseTensor& input,
                      pten::DenseTensor* output,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      bool reduce_all) {
  output->mutable_data<OutT>();

  if (reduce_all) {
    // Flatten and reduce 1-D tensor
    auto x = EigenVector<OutT>::Flatten(input);
    auto out = EigenScalar<OutT>::From(*output);
    auto& dev = *dev_ctx.eigen_device();
    auto reduce_dim = Eigen::array<int, 1>({{0}});

    Functor functor;
    functor(dev, &x, &out, reduce_dim);
  } else {
    int ndim = input.dims().size();
    int rdim = dims.size();
    if (ndim > 6) {
      HandleLargeDim<DeviceContext, OutT, Functor>(
          dev_ctx, input, output, dims, keep_dim);

    } else {
      HANDLE_REDUCE_DIM(6, 5);
      HANDLE_REDUCE_DIM(6, 4);
      HANDLE_REDUCE_DIM(6, 3);
      HANDLE_REDUCE_DIM(6, 2);
      HANDLE_REDUCE_DIM(6, 1);
      HANDLE_REDUCE_DIM(5, 4);
      HANDLE_REDUCE_DIM(5, 3);
      HANDLE_REDUCE_DIM(5, 2);
      HANDLE_REDUCE_DIM(5, 1);
      HANDLE_REDUCE_DIM(4, 3);
      HANDLE_REDUCE_DIM(4, 2);
      HANDLE_REDUCE_DIM(4, 1);
      HANDLE_REDUCE_DIM(3, 2);
      HANDLE_REDUCE_DIM(3, 1);
      HANDLE_REDUCE_DIM(2, 1);
      HANDLE_REDUCE_DIM(1, 1);
    }
  }
}

template <typename DeviceContext, typename T, typename Functor>
void Reduce(const DeviceContext& dev_ctx,
            const DenseTensor& x,
            bool reduce_all,
            const std::vector<int64_t>& dims,
            bool keep_dim,
            DataType out_dtype,
            DenseTensor* out) {
  // If the dims has full dim, set the reduce_all is True
  const int& input_dim_size = x.dims().size();
  std::set<int> dims_set(dims.begin(), dims.end());
  bool full_dim = true;
  for (int i = 0; i < input_dim_size; ++i) {
    if (dims_set.find(i) == dims_set.end() &&
        dims_set.find(i - input_dim_size) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || full_dim);

  // no need to cast dtype
  if (out_dtype == pten::DataType::UNDEFINED || out_dtype == x.dtype()) {
    if (out_dtype == pten::DataType::UNDEFINED) {
      out_dtype = x.dtype();
    }
    // do reduce sum
    PD_VISIT_ALL_TYPES(
        out_dtype, "ReduceKernelImpl", ([&] {
          pten::ReduceKernelImpl<DeviceContext, T, data_t, Functor>(
              dev_ctx, x, out, dims, keep_dim, reduce_all);
        }));
  } else {
    pten::DenseTensor tmp_tensor = pten::DenseTensor(
        pten::make_intrusive<paddle::experimental::SharedStorage>(x.place()),
        pten::DenseTensorMeta(out_dtype, x.dims(), x.layout()));

    // cast x tensor to out_dtype
    pten::CastKernel<T, DeviceContext>(dev_ctx, x, out_dtype, &tmp_tensor);

    // do reduce sum
    PD_VISIT_ALL_TYPES(
        out_dtype, "ReduceKernelImpl", ([&] {
          pten::ReduceKernelImpl<DeviceContext, T, data_t, Functor>(
              dev_ctx, tmp_tensor, out, dims, keep_dim, reduce_all);
        }));
  }
}

}  // namespace pten
