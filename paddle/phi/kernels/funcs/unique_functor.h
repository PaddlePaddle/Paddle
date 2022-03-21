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
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

template <typename Context, typename InT>
struct UniqueOpFunctor {
  const Context& context_;
  DenseTensor* out_;
  DenseTensor* index_;
  const DenseTensor* in_;
  DenseTensor* count_;

  UniqueOpFunctor(const Context& context,
                  DenseTensor* out,
                  DenseTensor* index,
                  const DenseTensor* in,
                  DenseTensor* count = nullptr)
      : context_(context), out_(out), index_(index), in_(in), count_(count) {}

  template <typename IndexT>
  void apply() const {
    auto* in_data = in_->data<InT>();
    auto* index_data = context_.template Alloc<IndexT>(index_);

    int64_t j = 0;

    // TODO(fangzeyang): Should optimize performance here.
    std::unordered_map<InT, int64_t> dict;
    std::vector<InT> uniq;

    PADDLE_ENFORCE_LT(
        in_->numel(),
        pow(2, 31),
        phi::errors::InvalidArgument(
            "The num of Input(X) elements should be less then INT_MAX, "
            "but received num is %d.",
            in_->numel()));

    for (auto i = 0; i < in_->numel(); i++) {
      auto it = dict.find(in_data[i]);
      if (it == dict.end()) {
        dict.emplace(std::make_pair(in_data[i], j));
        uniq.emplace_back(in_data[i]);
        index_data[i] = static_cast<IndexT>(j);
        j++;
      } else {
        index_data[i] = static_cast<IndexT>(it->second);
      }
    }

    if (count_ != nullptr) {
      // Resize the count tensor dims to allocate the memory
      count_->Resize(phi::make_ddim({static_cast<int64_t>(uniq.size())}));
      IndexT* count_data = context_.template Alloc<IndexT>(count_);
      // init count_data to 0
      memset(count_data, 0, uniq.size() * sizeof(IndexT));

      const auto& index_type = index_->dtype();
      bool index_type_match =
          index_type == DataType::INT32 || index_type == DataType::INT64;
      PADDLE_ENFORCE_EQ(
          index_type_match,
          true,
          phi::errors::InvalidArgument(
              "Index holds the wrong type, it holds %s, "
              "but desires to be %s or %s",
              paddle::framework::DataTypeToString(
                  paddle::framework::TransToProtoVarType(index_type)),
              paddle::framework::DataTypeToString(
                  paddle::framework::TransToProtoVarType(DataType::INT32)),
              paddle::framework::DataTypeToString(
                  paddle::framework::TransToProtoVarType(DataType::INT64))));

      if (index_type == DataType::INT32) {
        for (auto i = 0; i < in_->numel(); ++i) {
          const IndexT& index = index_data[i];
          count_data[static_cast<int32_t>(index)] += static_cast<IndexT>(1);
        }
      } else {
        for (auto i = 0; i < in_->numel(); ++i) {
          const IndexT& index = index_data[i];
          count_data[static_cast<int64_t>(index)] += static_cast<IndexT>(1);
        }
      }
    }

    out_->Resize(phi::make_ddim({static_cast<int64_t>(uniq.size())}));
    auto* out_data = context_.template Alloc<InT>(out_);
    std::memcpy(out_data, uniq.data(), uniq.size() * sizeof(InT));
  }
};

static std::vector<DenseTensor> Unbind(const DenseTensor& in) {
  int64_t size = in.dims()[0];
  std::vector<DenseTensor> tensors(size);
  for (int64_t i = 0; i < size; ++i) {
    tensors[i] = in.Slice(i, i + 1);
  }
  return tensors;
}

template <typename T>
static bool Equal(const DenseTensor& a, const DenseTensor& b) {
  if (a.numel() != b.numel()) {
    return false;
  }
  for (int64_t i = 0; i < a.numel(); ++i) {
    if (a.data<T>()[i] != b.data<T>()[i]) {
      return false;
    }
  }
  return true;
}

template <typename Context, typename InT, typename IndexT>
static void UniqueFlattendTensor(const Context& context,
                                 const DenseTensor& in,
                                 DenseTensor* out,
                                 DenseTensor* indices,
                                 DenseTensor* index,
                                 DenseTensor* count,
                                 bool return_index,
                                 bool return_inverse,
                                 bool return_counts) {
  const InT* in_data = in.data<InT>();
  std::set<InT> unique(in_data, in_data + in.numel());
  out->Resize(phi::make_ddim({static_cast<int64_t>(unique.size())}));
  auto* out_data = context.template Alloc<InT>(out);
  std::copy(unique.begin(), unique.end(), out_data);

  if (return_index) {
    indices->Resize(phi::make_ddim({out->numel()}));
    auto indices_data = context.template Alloc<IndexT>(indices);
    std::unordered_map<InT, IndexT> indices_map;
    indices_map.reserve(out->numel());
    for (int64_t i = 0; i < in.numel(); ++i) {
      if (indices_map.find(in_data[i]) != indices_map.end()) continue;
      indices_map[in_data[i]] = i;
    }
    for (int64_t i = 0; i < out->numel(); ++i) {
      indices_data[i] = indices_map[out_data[i]];
    }
  }

  if (return_inverse) {
    index->Resize(phi::make_ddim({in.numel()}));
    auto inverse_data = context.template Alloc<IndexT>(index);
    std::unordered_map<InT, IndexT> inverse_map;
    inverse_map.reserve(out->numel());
    for (int64_t i = 0; i < out->numel(); ++i) {
      inverse_map[out_data[i]] = i;
    }
    for (int64_t i = 0; i < in.numel(); ++i) {
      inverse_data[i] = inverse_map[in_data[i]];
    }
  }

  if (return_counts) {
    count->Resize(phi::make_ddim({out->numel()}));
    auto count_data = context.template Alloc<IndexT>(count);
    std::unordered_map<InT, IndexT> counts_map;
    counts_map.reserve(out->numel());
    for (int64_t i = 0; i < out->numel(); ++i) {
      counts_map[out_data[i]] = 0;
    }
    for (int64_t i = 0; i < in.numel(); i++) {
      counts_map[in_data[i]] += 1;
    }
    for (int64_t i = 0; i < out->numel(); i++) {
      count_data[i] = counts_map[out_data[i]];
    }
  }
}

template <typename Context, typename ForwardIt, typename InT, typename IndexT>
static ForwardIt UniqueDimImpl(const Context& context,
                               ForwardIt first,
                               ForwardIt last,
                               const std::vector<IndexT>& sorted_indices_vec,
                               std::vector<IndexT>* inverse_vec,
                               std::vector<IndexT>* counts_vec,
                               std::vector<IndexT>* indices_vec) {
  if (first == last) {
    return last;
  }

  (*inverse_vec)[sorted_indices_vec[0]] = 0;
  (*counts_vec)[0] = 1;
  (*indices_vec)[0] = sorted_indices_vec[0];

  ForwardIt begin = first;
  ForwardIt result = first;

  while (++first != last) {
    int64_t idx_first = std::distance(begin, first);
    int64_t idx_result = std::distance(begin, result);
    if (!Equal<InT>(*result, *first)) {
      if (++result != first) {
        *result = std::move(*first);
      }
      idx_result += 1;
      (*indices_vec)[idx_result] = sorted_indices_vec[idx_first];
    }
    (*inverse_vec)[sorted_indices_vec[idx_first]] = idx_result;
    (*counts_vec)[idx_result] += 1;
  }
  return ++result;
}

template <typename Context, typename InT, typename IndexT>
static void UniqueDim(const Context& context,
                      const DenseTensor& in,
                      DenseTensor* out,
                      DenseTensor* indices,
                      DenseTensor* index,
                      DenseTensor* count,
                      bool return_index,
                      bool return_inverse,
                      bool return_counts,
                      int axis) {
  // transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
  std::vector<int> permute(in.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis] = 0;
  permute[0] = axis;
  std::vector<int64_t> in_trans_dims_vec(phi::vectorize(in.dims()));
  in_trans_dims_vec[axis] = in.dims()[0];
  in_trans_dims_vec[0] = in.dims()[axis];
  DenseTensor in_trans;
  phi::DDim in_trans_dims = phi::make_ddim(in_trans_dims_vec);
  in_trans.Resize(in_trans_dims);
  context.template Alloc<InT>(&in_trans);
  TransCompute<Context, InT>(in.dims().size(), context, in, &in_trans, permute);
  // reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  phi::DDim in_trans_flat_dims = phi::flatten_to_2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  // sort indices
  std::vector<IndexT> sorted_indices_vec(in_trans.dims()[0]);
  std::iota(sorted_indices_vec.begin(), sorted_indices_vec.end(), 0);
  int64_t col = in_trans.dims()[1];
  const InT* in_trans_data = in_trans.data<InT>();
  std::sort(sorted_indices_vec.begin(),
            sorted_indices_vec.end(),
            [&](int64_t a, int64_t b) -> bool {
              for (int64_t i = 0; i < col; ++i) {
                InT lhs = in_trans_data[i + a * col];
                InT rhs = in_trans_data[i + b * col];
                if (lhs < rhs) {
                  return true;
                } else if (lhs > rhs) {
                  return false;
                }
              }
              return false;
            });

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

  std::vector<DenseTensor> input_unbind = Unbind(input_sorted);
  std::vector<IndexT> inverse_vec(sorted_indices_vec.size(), 0);
  std::vector<IndexT> counts_vec(sorted_indices_vec.size(), 0);
  std::vector<IndexT> indices_vec(sorted_indices_vec.size(), 0);
  auto last = UniqueDimImpl<Context, std::vector<DenseTensor>::iterator, InT>(
      context,
      input_unbind.begin(),
      input_unbind.end(),
      sorted_indices_vec,
      &inverse_vec,
      &counts_vec,
      &indices_vec);
  input_unbind.erase(last, input_unbind.end());
  counts_vec.erase(counts_vec.begin() + input_unbind.size(), counts_vec.end());
  indices_vec.erase(indices_vec.begin() + input_unbind.size(),
                    indices_vec.end());

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
  TransCompute<Context, InT>(
      out_trans.dims().size(), context, out_trans, out, permute);

  if (return_inverse) {
    paddle::framework::TensorFromVector(inverse_vec, context, index);
  }

  if (return_counts) {
    paddle::framework::TensorFromVector(counts_vec, context, count);
  }

  if (return_index) {
    paddle::framework::TensorFromVector(indices_vec, context, indices);
  }
}

template <typename Context, typename InT>
struct UniqueFlattendTensorFunctor {
  const Context& ctx_; /*  */
  const DenseTensor& in_;
  DenseTensor* out_;
  DenseTensor* indices_;
  DenseTensor* index_;
  DenseTensor* count_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueFlattendTensorFunctor(const Context& context,
                              const DenseTensor& in,
                              DenseTensor* out,
                              DenseTensor* indices,
                              DenseTensor* index,
                              DenseTensor* count,
                              bool return_index,
                              bool return_inverse,
                              bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        indices_(indices),
        index_(index),
        count_(count),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueFlattendTensor<Context, InT, IndexT>(ctx_,
                                               in_,
                                               out_,
                                               indices_,
                                               index_,
                                               count_,
                                               return_index_,
                                               return_inverse_,
                                               return_counts_);
  }
};

template <typename Context, typename InT>
struct UniqueDimFunctor {
  const Context& ctx_;
  const DenseTensor& in_;
  DenseTensor* out_;
  DenseTensor* indices_;
  DenseTensor* index_;
  DenseTensor* count_;
  const int axis_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueDimFunctor(const Context& context,
                   const DenseTensor& in,
                   DenseTensor* out,
                   DenseTensor* indices,
                   DenseTensor* index,
                   DenseTensor* count,
                   const int axis,
                   bool return_index,
                   bool return_inverse,
                   bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        indices_(indices),
        index_(index),
        count_(count),
        axis_(axis),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueDim<Context, InT, IndexT>(ctx_,
                                    in_,
                                    out_,
                                    indices_,
                                    index_,
                                    count_,
                                    return_index_,
                                    return_inverse_,
                                    return_counts_,
                                    axis_);
  }
};

}  // namespace funcs
}  // namespace phi
