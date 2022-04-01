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

#include "paddle/phi/kernels/unique_kernel.h"

#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <iostream>
#include <vector>
#include "paddle/fluid/framework/tensor_util.h"  // TensorToVector()
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/unique_functor.h"

namespace phi {

// Binary function 'less than'
template <typename InT>
struct LessThan {
  int col;
  const InT* in_trans_data;

  LessThan(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  }
};

// Binary function 'equal_to'
template <typename InT>
struct BinaryEqual {
  int64_t col;
  const InT* in_trans_data;

  BinaryEqual(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  }
};

// Binary function 'not_equal_to'
template <typename InT>
struct BinaryNotEqual {
  int64_t col;
  const InT* in_trans_data;

  BinaryNotEqual(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs != rhs) {
        return true;
      }
    }
    return false;
  }
};

// index_select() function for DenseTensor
template <typename Context, typename InT, typename IndexT>
void IndexSelect(const Context& context,
                 const DenseTensor& input,
                 const DenseTensor& index,
                 DenseTensor* output,
                 int dim) {
  auto input_dim = input.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();

  auto slice_size = 1;
  for (auto i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }

  auto input_width = slice_size * input_dim[dim];
  auto output_width = slice_size * output_dim[dim];

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  auto index_size = index.dims()[0];

  std::vector<InT> input_vec;
  std::vector<IndexT> index_vec;
  paddle::framework::TensorToVector(input, context, &input_vec);
  paddle::framework::TensorToVector(index, context, &index_vec);
  std::vector<InT> out_vec(output->numel());

  for (int i = 0; i < index_size; i++) {
    PADDLE_ENFORCE_GE(
        index_vec[i],
        0,
        phi::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim],
            index_vec[i]));
    PADDLE_ENFORCE_LT(
        index_vec[i],
        input_dim[dim],
        phi::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim],
            index_vec[i]));
  }

  for (auto i = 0; i < outer_nums; i++) {
    auto input_start_offset = i * input_width;
    auto output_start_offset = i * output_width;

    for (auto j = 0; j < index_size; j++) {
      IndexT index_value = index_vec[j];
      for (auto k = 0; k < slice_size; k++) {
        out_vec[output_start_offset + j * slice_size + k] =
            input_vec[input_start_offset + index_value * slice_size + k];
      }
    }
  }
  context.template Alloc<IndexT>(output);
  paddle::framework::TensorFromVector(out_vec, context, output);
  output->Resize(output_dim);
}

// The core logic of computing Unique for a flattend DenseTensor
template <typename Context,
          typename InT,
          typename IndexT,
          typename equal_T,
          typename not_equal_T>
static void UniqueFlattendCUDATensor(const Context& context,
                                     const DenseTensor& in,
                                     DenseTensor* out,
                                     DenseTensor* indices,
                                     DenseTensor* index,
                                     DenseTensor* counts,
                                     bool return_index,
                                     bool return_inverse,
                                     bool return_counts,
                                     equal_T equal,
                                     not_equal_T not_equal,
                                     int64_t num_input) {
  // 0. Prepration
  DenseTensor in_hat;
  phi::Copy(context, in, context.GetPlace(), false, &in_hat);
  auto* in_data_hat = context.template Alloc<InT>(&in_hat);

  indices->Resize(phi::make_ddim({num_input}));
  auto* indices_data = context.template Alloc<IndexT>(indices);

  thrust::sequence(thrust::device, indices_data, indices_data + num_input);
  thrust::sort_by_key(
      thrust::device, in_data_hat, in_data_hat + num_input, indices_data);

  // 1. Calculate op result: 'out'
  DenseTensor range;
  range.Resize(phi::make_ddim({num_input + 1}));
  auto* range_data_ptr = context.template Alloc<IndexT>(&range);
  thrust::sequence(
      thrust::device, range_data_ptr, range_data_ptr + num_input + 1);
  phi::Copy(context, in_hat, context.GetPlace(), false, out);
  int num_out;
  auto out_data = context.template Alloc<InT>(out);
  num_out =
      thrust::unique_by_key(
          thrust::device, out_data, out_data + num_input, range_data_ptr, equal)
          .first -
      out_data;
  out->Resize(phi::make_ddim({num_out}));

  // 3. Calculate inverse index: 'inverse'
  if (return_inverse) {
    index->Resize(phi::make_ddim({num_input}));
    auto* inverse_data = context.template Alloc<IndexT>(index);
    DenseTensor inv_loc;
    inv_loc.Resize(phi::make_ddim({num_input}));
    auto inv_loc_data_ptr = context.template Alloc<IndexT>(&inv_loc);
    thrust::adjacent_difference(thrust::device,
                                in_data_hat,
                                in_data_hat + num_input,
                                inv_loc_data_ptr,
                                not_equal);
    thrust::device_ptr<IndexT> inv_loc_data_dev(inv_loc_data_ptr);
    inv_loc_data_dev[0] = 0;  // without device_ptr, segmentation fault
    thrust::inclusive_scan(thrust::device,
                           inv_loc_data_ptr,
                           inv_loc_data_ptr + num_input,
                           inv_loc_data_ptr);
    thrust::scatter(thrust::device,
                    inv_loc_data_ptr,
                    inv_loc_data_ptr + num_input,
                    indices_data,
                    inverse_data);
  }

  // 2. Calculate sorted index: 'indices'
  if (return_index) {
    DenseTensor tmp_indices;
    tmp_indices.Resize(phi::make_ddim({num_input}));
    auto* tmp_indices_data_ptr = context.template Alloc<IndexT>(&tmp_indices);
    thrust::copy(thrust::device,
                 in_data_hat,
                 in_data_hat + num_input,
                 tmp_indices_data_ptr);
    thrust::unique_by_key(thrust::device,
                          tmp_indices_data_ptr,
                          tmp_indices_data_ptr + num_input,
                          indices_data,
                          equal);
    indices->Resize(phi::make_ddim({num_out}));
  }

  // 4. Calculate 'counts'
  if (return_counts) {
    counts->Resize(phi::make_ddim({num_out}));
    auto count_data = context.template Alloc<IndexT>(counts);
    // init 'count_data' as 0
    thrust::fill(thrust::device, count_data, count_data + num_out, 0);
    thrust::device_ptr<IndexT> range_data_ptr_dev(range_data_ptr);
    range_data_ptr_dev[num_out] = num_input;
    thrust::adjacent_difference(thrust::device,
                                range_data_ptr + 1,
                                range_data_ptr + num_out + 1,
                                count_data);
  }
}

// The logic of compute unique with axis required, it's a little different
// from above function
template <typename Context,
          typename InT,
          typename IndexT,
          typename equal_T,
          typename not_equal_T>
static void ComputeUniqueDims(const Context& context,
                              DenseTensor* sorted_indices,
                              IndexT* sorted_indices_data,
                              DenseTensor* out,
                              DenseTensor* inverse,
                              DenseTensor* counts,
                              bool return_index,
                              bool return_inverse,
                              bool return_counts,
                              equal_T equal,
                              not_equal_T not_equal,
                              int64_t row) {
  // 1. inverse indices: 'inverse'
  inverse->Resize(phi::make_ddim({row}));
  auto* inverse_data = context.template Alloc<IndexT>(inverse);
  DenseTensor inv_loc;
  inv_loc.Resize(phi::make_ddim({row}));
  auto inv_loc_data_ptr = context.template Alloc<IndexT>(&inv_loc);
  thrust::adjacent_difference(thrust::device,
                              sorted_indices_data,
                              sorted_indices_data + row,
                              inv_loc_data_ptr,
                              not_equal);
  thrust::device_ptr<IndexT> inv_loc_data_dev(inv_loc_data_ptr);
  inv_loc_data_dev[0] = 0;
  thrust::inclusive_scan(thrust::device,
                         inv_loc_data_ptr,
                         inv_loc_data_ptr + row,
                         inv_loc_data_ptr);
  thrust::scatter(thrust::device,
                  inv_loc_data_ptr,
                  inv_loc_data_ptr + row,
                  sorted_indices_data,
                  inverse_data);

  // 2. sorted indices
  DenseTensor range;
  range.Resize(phi::make_ddim({row + 1}));
  auto range_data_ptr = context.template Alloc<IndexT>(&range);
  thrust::sequence(thrust::device, range_data_ptr, range_data_ptr + row + 1);
  int num_out;
  num_out = thrust::unique_by_key(thrust::device,
                                  sorted_indices_data,
                                  sorted_indices_data + row,
                                  range_data_ptr,
                                  equal)
                .first -
            sorted_indices_data;
  thrust::device_ptr<IndexT> range_data_ptr_dev(range_data_ptr);
  range_data_ptr_dev[num_out] = row;
  sorted_indices->Resize(phi::make_ddim({num_out}));

  // 3. counts: 'counts'
  counts->Resize(phi::make_ddim({num_out}));
  auto* count_data = context.template Alloc<IndexT>(counts);
  thrust::fill(thrust::device, count_data, count_data + row, 0);
  thrust::adjacent_difference(
      thrust::device, range_data_ptr + 1, range_data_ptr + row + 1, count_data);
}

// Calculate unique when 'axis' is set
template <typename Context, typename InT, typename IndexT>
static void UniqueDimsCUDATensor(const Context& context,
                                 const DenseTensor& in,
                                 DenseTensor* out,
                                 DenseTensor* indices,
                                 DenseTensor* index,
                                 DenseTensor* counts,
                                 bool return_index,
                                 bool return_inverse,
                                 bool return_counts,
                                 int axis) {
  // 1. Transpose & reshape
  // Transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
  std::vector<int> permute(in.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis] = 0;
  permute[0] = axis;
  std::vector<int64_t> in_trans_dims_vec(phi::vectorize(in.dims()));
  in_trans_dims_vec[axis] = in.dims()[0];
  in_trans_dims_vec[0] = in.dims()[axis];
  DenseTensor in_trans;
  auto in_trans_dims = phi::make_ddim(in_trans_dims_vec);
  in_trans.Resize(in_trans_dims);
  context.template Alloc<InT>(&in_trans);
  phi::funcs::TransCompute<Context, InT>(
      in.dims().size(),  // num of dims
      context,           // device
      in,                // original DenseTensor
      &in_trans,         // DenseTensor after reshape
      permute);          // index of axis

  // Reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  auto in_trans_flat_dims = phi::flatten_to_2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  // now 'in_trans' is 2D
  int64_t col = in_trans.dims()[1];
  int64_t row = in_trans.dims()[0];
  const InT* in_trans_data = in_trans.data<InT>();

  indices->Resize(phi::make_ddim({row}));
  auto* sorted_indices_data = context.template Alloc<IndexT>(indices);

  // 2. Calculate 'indices', 'inverse', 'counts'
  // Init index and sort
  thrust::sequence(
      thrust::device, sorted_indices_data, sorted_indices_data + row);
  thrust::sort(thrust::device,
               sorted_indices_data,
               sorted_indices_data + row,
               LessThan<InT>(col, in_trans_data));
  ComputeUniqueDims<Context, InT, IndexT>(
      context,
      indices,
      sorted_indices_data,
      out,
      index,
      counts,
      return_index,
      return_inverse,
      return_counts,
      BinaryEqual<InT>(col, in_trans_data),
      BinaryNotEqual<InT>(col, in_trans_data),
      row);

  // 3. Select indices and reshape back to get 'out'
  DenseTensor out_trans;
  std::vector<int64_t> out_trans_dims_vec = in_trans_dims_vec;
  out_trans_dims_vec[0] = indices->numel();
  out_trans.Resize(phi::make_ddim(out_trans_dims_vec));
  context.template Alloc<InT>(&out_trans);

  IndexSelect<Context, InT, IndexT>(context, in_trans, *indices, &out_trans, 0);

  std::swap(out_trans_dims_vec[0], out_trans_dims_vec[axis]);
  out->Resize(phi::make_ddim(out_trans_dims_vec));
  context.template Alloc<InT>(out);
  std::vector<DenseTensor> out_trans_unbind = phi::funcs::Unbind(out_trans);
  phi::funcs::ConcatFunctor<Context, InT> concat_functor;
  concat_functor(context, out_trans_unbind, 0, &out_trans);
  phi::funcs::TransCompute<Context, InT>(
      out_trans.dims().size(), context, out_trans, out, permute);
}

// functor for processing a flattend DenseTensor
template <typename Context, typename InT>
struct UniqueFlattendCUDAFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  DenseTensor* indices_;
  DenseTensor* index_;
  DenseTensor* counts_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueFlattendCUDAFunctor(const Context& context,
                            const DenseTensor& in,
                            DenseTensor* out,
                            DenseTensor* indices,
                            DenseTensor* index,
                            DenseTensor* counts,
                            bool return_index,
                            bool return_inverse,
                            bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        indices_(indices),
        index_(index),
        counts_(counts),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueFlattendCUDATensor<Context, InT, IndexT>(ctx_,
                                                   in_,
                                                   out_,
                                                   indices_,
                                                   index_,
                                                   counts_,
                                                   return_index_,
                                                   return_inverse_,
                                                   return_counts_,
                                                   thrust::equal_to<InT>(),
                                                   thrust::not_equal_to<InT>(),
                                                   in_.numel());
  }
};

// functor for processing a multi-dimentional DenseTensor
template <typename Context, typename InT>
struct UniqueDimsCUDAFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  DenseTensor* indices_;
  DenseTensor* index_;
  DenseTensor* counts_;
  const int axis_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueDimsCUDAFunctor(const Context& context,
                        const DenseTensor& in,
                        DenseTensor* out,
                        DenseTensor* indices,
                        DenseTensor* index,
                        DenseTensor* counts,
                        const int axis,
                        bool return_index,
                        bool return_inverse,
                        bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        indices_(indices),
        index_(index),
        counts_(counts),
        axis_(axis),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueDimsCUDATensor<Context, InT, IndexT>(ctx_,
                                               in_,
                                               out_,
                                               indices_,
                                               index_,
                                               counts_,
                                               return_index_,
                                               return_inverse_,
                                               return_counts_,
                                               axis_);
  }
};

template <typename T, typename Context>
void UniqueRawKernel(const Context& context,
                     const DenseTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     bool is_sorted,
                     DenseTensor* out,
                     DenseTensor* indices,
                     DenseTensor* index,
                     DenseTensor* counts) {
  if (dtype == phi::DataType::INT32) {
    PADDLE_ENFORCE_LE(
        x.numel() + 1,
        INT_MAX,
        phi::errors::InvalidArgument(
            "The number of elements in Input(X) should be less than or "
            "equal to INT_MAX, but received num is %d. Please set `dtype` to "
            "int64.",
            x.numel()));
  }
  // if 'axis' is not required, flatten the DenseTensor.
  if (axis.empty()) {
    phi::VisitDataTypeTiny(
        dtype,
        UniqueFlattendCUDAFunctor<Context, T>(context,
                                              x,
                                              out,
                                              indices,
                                              index,
                                              counts,
                                              return_index,
                                              return_inverse,
                                              return_counts));
  } else {
    // 'axis' is required.
    int axis_value = axis[0];
    phi::VisitDataTypeTiny(dtype,
                           UniqueDimsCUDAFunctor<Context, T>(context,
                                                             x,
                                                             out,
                                                             indices,
                                                             index,
                                                             counts,
                                                             axis_value,
                                                             return_index,
                                                             return_inverse,
                                                             return_counts));
  }
}

template <typename T, typename Context>
void UniqueKernel(const Context& context,
                  const DenseTensor& x,
                  bool return_index,
                  bool return_inverse,
                  bool return_counts,
                  const std::vector<int>& axis,
                  DataType dtype,
                  DenseTensor* out,
                  DenseTensor* indices,
                  DenseTensor* index,
                  DenseTensor* counts) {
  bool is_sorted = true;
  UniqueRawKernel<T, Context>(context,
                              x,
                              return_index,
                              return_inverse,
                              return_counts,
                              axis,
                              dtype,
                              is_sorted,
                              out,
                              indices,
                              index,
                              counts);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    unique, GPU, ALL_LAYOUT, phi::UniqueKernel, float, double, int64_t, int) {}

PD_REGISTER_KERNEL(unique_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::UniqueRawKernel,
                   float,
                   double,
                   int64_t,
                   int) {}
