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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/unique_functor.h"

namespace phi {

template <typename InT, typename IndexT, typename Context>
static void UniqueConsecutiveFlattenedTensor(const Context& context,
                                             const DenseTensor& in,
                                             DenseTensor* out,
                                             bool return_inverse,
                                             bool return_counts,
                                             DenseTensor* inverse,
                                             DenseTensor* count) {
  const InT* in_data = in.data<InT>();
  std::vector<InT> out_vec(in.numel());
  std::vector<IndexT> inverse_vec(in.numel());
  std::vector<IndexT> counts_vec(in.numel());
  memcpy(out_vec.data(), in_data, in.numel() * sizeof(InT));
  InT* p = out_vec.data();
  int64_t last = 0;
  IndexT* q = counts_vec.data();
  for (int64_t i = 0; i < in.numel(); i++) {
    if (in_data[i] != *p) {
      *(++p) = in_data[i];
      if (return_counts) {
        *(q++) = i - last;
        last = i;
      }
    }
    if (return_inverse) {
      inverse_vec[i] = p - out_vec.data();
    }
  }

  bool is_empty = in.numel() == 0;
  int64_t output_size = is_empty ? 0 : (p - out_vec.data() + 1);

  if (return_counts) {
    if (!is_empty) *q = in.numel() - last;
    counts_vec.resize(output_size);
  }
  out_vec.resize(output_size);

  out->Resize(phi::make_ddim({output_size}));
  auto* out_data = context.template Alloc<InT>(out);
  std::copy(out_vec.begin(), out_vec.end(), out_data);

  if (return_inverse) {
    inverse->Resize(phi::make_ddim({in.numel()}));
    auto* inverse_data = context.template Alloc<IndexT>(inverse);
    std::copy(inverse_vec.begin(), inverse_vec.end(), inverse_data);
  }

  if (return_counts) {
    count->Resize(phi::make_ddim({out->numel()}));
    auto* counts_data = context.template Alloc<IndexT>(count);
    std::copy(counts_vec.begin(), counts_vec.end(), counts_data);
  }
}

template <typename Context, typename InT>
struct UniqueConsecutiveFlattenedTensorFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  const bool return_inverse_;
  const bool return_counts_;
  DenseTensor* inverse_;
  DenseTensor* count_;

  UniqueConsecutiveFlattenedTensorFunctor(const Context& context,
                                          const DenseTensor& in,
                                          DenseTensor* out,
                                          bool return_inverse,
                                          bool return_counts,
                                          DenseTensor* inverse,
                                          DenseTensor* count)
      : ctx_(context),
        in_(in),
        out_(out),
        return_inverse_(return_inverse),
        return_counts_(return_counts),
        inverse_(inverse),
        count_(count) {}

  template <typename IndexT>
  void apply() const {
    UniqueConsecutiveFlattenedTensor<InT, IndexT, Context>(
        ctx_, in_, out_, return_inverse_, return_counts_, inverse_, count_);
  }
};

template <typename Context, class ForwardIt, typename InT, typename IndexT>
static ForwardIt UniqueConsecutiveDimImpl(
    const Context& context UNUSED,
    ForwardIt first,
    ForwardIt last,
    const std::vector<IndexT>& sorted_indices_vec,
    std::vector<IndexT>* inverse_vec,
    std::vector<IndexT>* counts_vec) {
  if (first == last) {
    return last;
  }

  (*inverse_vec)[sorted_indices_vec[0]] = 0;
  (*counts_vec)[0] = 1;

  ForwardIt begin = first;
  ForwardIt result = first;

  while (++first != last) {
    int64_t idx_first = std::distance(begin, first);
    int64_t idx_result = std::distance(begin, result);
    if (!phi::funcs::Equal<InT>(*result, *first)) {
      if (++result != first) {
        *result = std::move(*first);
      }
      idx_result += 1;
    }
    (*inverse_vec)[sorted_indices_vec[idx_first]] = idx_result;
    (*counts_vec)[idx_result] += 1;
  }
  return ++result;
}

template <typename Context, typename InT, typename IndexT>
static void UniqueConsecutiveDim(const Context& context,
                                 const DenseTensor& in,
                                 DenseTensor* out,
                                 bool return_inverse,
                                 bool return_counts,
                                 int axis,
                                 DenseTensor* inverse,
                                 DenseTensor* count) {
  // transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
  std::vector<int> permute(in.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis] = 0;
  permute[0] = axis;
  std::vector<int64_t> in_trans_dims_vec(phi::vectorize(in.dims()));
  in_trans_dims_vec[axis] = in.dims()[0];
  in_trans_dims_vec[0] = in.dims()[axis];
  DenseTensor in_trans;
  DDim in_trans_dims = phi::make_ddim(in_trans_dims_vec);
  in_trans.Resize(in_trans_dims);
  context.template Alloc<InT>(&in_trans);
  phi::funcs::TransCompute<Context, InT>(
      in.dims().size(), context, in, &in_trans, permute);
  // reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  DDim in_trans_flat_dims = phi::flatten_to_2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  std::vector<IndexT> sorted_indices_vec(in_trans.dims()[0]);
  std::iota(sorted_indices_vec.begin(), sorted_indices_vec.end(), 0);
  int64_t col = in_trans.dims()[1];
  const InT* in_trans_data = in_trans.data<InT>();

  // sort tensor according to indices
  DenseTensor input_sorted;
  input_sorted.Resize(in_trans_dims);
  context.template Alloc<InT>(&input_sorted);
  InT* input_sorted_data = input_sorted.data<InT>();
  for (size_t i = 0; i < sorted_indices_vec.size(); ++i) {
    memcpy(input_sorted_data + i * col,
           in_trans_data + static_cast<int64_t>(sorted_indices_vec[i]) * col,
           col * sizeof(InT));
  }
  std::vector<DenseTensor> input_unbind = phi::funcs::Unbind(input_sorted);
  std::vector<IndexT> inverse_vec(sorted_indices_vec.size(), 0);
  std::vector<IndexT> counts_vec(sorted_indices_vec.size(), 0);
  auto last = UniqueConsecutiveDimImpl<Context,
                                       std::vector<DenseTensor>::iterator,
                                       InT>(context,
                                            input_unbind.begin(),
                                            input_unbind.end(),
                                            sorted_indices_vec,
                                            &inverse_vec,
                                            &counts_vec);
  input_unbind.erase(last, input_unbind.end());
  counts_vec.erase(counts_vec.begin() + input_unbind.size(), counts_vec.end());

  phi::funcs::ConcatFunctor<Context, InT> concat_functor;
  DenseTensor out_trans;
  std::vector<int64_t> out_trans_dims_vec = in_trans_dims_vec;
  out_trans_dims_vec[0] = input_unbind.size();
  out_trans.Resize(phi::make_ddim(out_trans_dims_vec));
  context.template Alloc<InT>(&out_trans);
  std::swap(out_trans_dims_vec[0], out_trans_dims_vec[axis]);
  out->Resize(phi::make_ddim(out_trans_dims_vec));
  context.template Alloc<InT>(out);
  concat_functor(context, input_unbind, 0, &out_trans);
  phi::funcs::TransCompute<Context, InT>(
      out_trans.dims().size(), context, out_trans, out, permute);
  if (return_inverse) {
    phi::TensorFromVector(inverse_vec, context, inverse);
  }
  if (return_counts) {
    phi::TensorFromVector(counts_vec, context, count);
  }
}

template <typename Context, typename InT>
struct UniqueConsecutiveDimFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  const int axis_;
  const bool return_inverse_;
  const bool return_counts_;
  DenseTensor* inverse_;
  DenseTensor* count_;

  UniqueConsecutiveDimFunctor(const Context& context,
                              const DenseTensor& in,
                              DenseTensor* out,
                              const int axis,
                              bool return_inverse,
                              bool return_counts,
                              DenseTensor* inverse,
                              DenseTensor* count)
      : ctx_(context),
        in_(in),
        out_(out),
        axis_(axis),
        return_inverse_(return_inverse),
        return_counts_(return_counts),
        inverse_(inverse),
        count_(count) {}

  template <typename IndexT>
  void apply() const {
    UniqueConsecutiveDim<Context, InT, IndexT>(ctx_,
                                               in_,
                                               out_,
                                               return_inverse_,
                                               return_counts_,
                                               axis_,
                                               inverse_,
                                               count_);
  }
};

}  // namespace phi
