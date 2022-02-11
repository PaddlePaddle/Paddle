/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/operators/unique_op.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
template <typename InT, typename IndexT>
static void UniqueConsecutiveFlattendTensor(
    const framework::ExecutionContext& context, const framework::Tensor& in,
    framework::Tensor* out, bool return_inverse, bool return_counts) {
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

  int64_t output_size = p - out_vec.data() + 1;
  if (return_counts) {
    *q = in.numel() - last;
    counts_vec.resize(output_size);
  }
  out_vec.resize(output_size);

  out->Resize(framework::make_ddim({output_size}));
  auto* out_data = out->mutable_data<InT>(context.GetPlace());
  std::copy(out_vec.begin(), out_vec.end(), out_data);

  if (return_inverse) {
    auto* inverse = context.Output<framework::Tensor>("Index");
    inverse->Resize(framework::make_ddim({in.numel()}));
    auto* inverse_data = inverse->mutable_data<IndexT>(context.GetPlace());
    std::copy(inverse_vec.begin(), inverse_vec.end(), inverse_data);
  }

  if (return_counts) {
    auto* count = context.Output<framework::Tensor>("Counts");
    count->Resize(framework::make_ddim({out->numel()}));
    auto* counts_data = count->mutable_data<IndexT>(context.GetPlace());
    std::copy(counts_vec.begin(), counts_vec.end(), counts_data);
  }
}

template <class ForwardIt, typename InT, typename IndexT>
static ForwardIt UniqueConsecutiveDimImpl(
    const framework::ExecutionContext& context, ForwardIt first, ForwardIt last,
    const std::vector<IndexT>& sorted_indices_vec,
    std::vector<IndexT>* inverse_vec, std::vector<IndexT>* counts_vec) {
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
    if (!Equal<InT>(*result, *first)) {
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

template <typename DeviceContext, typename InT, typename IndexT>
static void UniqueConsecutiveDim(const framework::ExecutionContext& context,
                                 const framework::Tensor& in,
                                 framework::Tensor* out, bool return_inverse,
                                 bool return_counts, int axis) {
  // transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
  std::vector<int> permute(in.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis] = 0;
  permute[0] = axis;
  std::vector<int64_t> in_trans_dims_vec(framework::vectorize(in.dims()));
  in_trans_dims_vec[axis] = in.dims()[0];
  in_trans_dims_vec[0] = in.dims()[axis];
  framework::Tensor in_trans;
  framework::DDim in_trans_dims = framework::make_ddim(in_trans_dims_vec);
  in_trans.Resize(in_trans_dims);
  in_trans.mutable_data<InT>(context.GetPlace());
  auto& dev_ctx = context.template device_context<DeviceContext>();
  TransCompute<DeviceContext, InT>(in.dims().size(), dev_ctx, in, &in_trans,
                                   permute);
  // reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  framework::DDim in_trans_flat_dims =
      framework::flatten_to_2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  std::vector<IndexT> sorted_indices_vec(in_trans.dims()[0]);
  std::iota(sorted_indices_vec.begin(), sorted_indices_vec.end(), 0);
  int64_t col = in_trans.dims()[1];
  const InT* in_trans_data = in_trans.data<InT>();

  // sort tensor according to indices
  framework::Tensor input_sorted;
  input_sorted.Resize(in_trans_dims);
  input_sorted.mutable_data<InT>(context.GetPlace());
  InT* input_sorted_data = input_sorted.data<InT>();
  for (size_t i = 0; i < sorted_indices_vec.size(); ++i) {
    memcpy(input_sorted_data + i * col,
           in_trans_data + static_cast<int64_t>(sorted_indices_vec[i]) * col,
           col * sizeof(InT));
  }
  std::vector<framework::Tensor> input_unbind = Unbind(input_sorted);
  std::vector<IndexT> inverse_vec(sorted_indices_vec.size(), 0);
  std::vector<IndexT> counts_vec(sorted_indices_vec.size(), 0);
  auto last =
      UniqueConsecutiveDimImpl<std::vector<framework::Tensor>::iterator, InT>(
          context, input_unbind.begin(), input_unbind.end(), sorted_indices_vec,
          &inverse_vec, &counts_vec);
  input_unbind.erase(last, input_unbind.end());
  counts_vec.erase(counts_vec.begin() + input_unbind.size(), counts_vec.end());

  math::ConcatFunctor<DeviceContext, InT> concat_functor;
  framework::Tensor out_trans;
  std::vector<int64_t> out_trans_dims_vec = in_trans_dims_vec;
  out_trans_dims_vec[0] = input_unbind.size();
  out_trans.Resize(framework::make_ddim(out_trans_dims_vec));
  out_trans.mutable_data<InT>(context.GetPlace());
  std::swap(out_trans_dims_vec[0], out_trans_dims_vec[axis]);
  out->Resize(framework::make_ddim(out_trans_dims_vec));
  out->mutable_data<InT>(context.GetPlace());
  concat_functor(dev_ctx, input_unbind, 0, &out_trans);
  TransCompute<DeviceContext, InT>(out_trans.dims().size(), dev_ctx, out_trans,
                                   out, permute);
  if (return_inverse) {
    auto* inverse = context.Output<framework::Tensor>("Index");
    framework::TensorFromVector(inverse_vec, context.device_context(), inverse);
  }
  if (return_counts) {
    auto* count = context.Output<framework::Tensor>("Counts");
    framework::TensorFromVector(counts_vec, context.device_context(), count);
  }
}

template <typename DeviceContext, typename InT>
struct UniqueConsecutiveFlattendTensorFunctor {
  const framework::ExecutionContext& ctx_;
  const framework::Tensor& in_;
  framework::Tensor* out_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueConsecutiveFlattendTensorFunctor(
      const framework::ExecutionContext& context, const framework::Tensor& in,
      framework::Tensor* out, bool return_inverse, bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueConsecutiveFlattendTensor<InT, IndexT>(
        ctx_, in_, out_, return_inverse_, return_counts_);
  }
};

template <typename DeviceContext, typename InT>
struct UniqueConsecutiveDimFunctor {
  const framework::ExecutionContext& ctx_;
  const framework::Tensor& in_;
  framework::Tensor* out_;
  const int axis_;
  const bool return_inverse_;
  const bool return_counts_;
  UniqueConsecutiveDimFunctor(const framework::ExecutionContext& context,
                              const framework::Tensor& in,
                              framework::Tensor* out, const int axis,
                              bool return_inverse, bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        axis_(axis),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueConsecutiveDim<DeviceContext, InT, IndexT>(
        ctx_, in_, out_, return_inverse_, return_counts_, axis_);
  }
};
template <typename DeviceContext, typename T>
class UniqueConsecutiveKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto data_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    if (data_type == framework::proto::VarType::INT32) {
      PADDLE_ENFORCE_LE(
          x->numel(), INT_MAX,
          platform::errors::InvalidArgument(
              "The number of elements in Input(X) should be less than or "
              "equal to INT_MAX, but received num is %d. Please set `dtype` to "
              "int64.",
              x->numel()));
    }
    std::vector<int> axis_vec = context.Attr<std::vector<int>>("axis");
    bool return_inverse = context.Attr<bool>("return_inverse");
    bool return_counts = context.Attr<bool>("return_counts");

    if (axis_vec.empty()) {
      framework::VisitDataTypeTiny(
          data_type, UniqueConsecutiveFlattendTensorFunctor<DeviceContext, T>(
                         context, *x, out, return_inverse, return_counts));
    } else {
      int axis = axis_vec[0];
      framework::VisitDataTypeTiny(
          data_type,
          UniqueConsecutiveDimFunctor<DeviceContext, T>(
              context, *x, out, axis, return_inverse, return_counts));
    }
  }
};
}  // namespace operators
}  // namespace paddle
