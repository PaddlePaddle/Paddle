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
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"
#include "paddle/fluid/prim/api/generated/prim_api/prim_api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace prim {
// We put some api like utils here
template <typename T>
Tensor empty(const paddle::experimental::IntArray& shape,
             paddle::experimental::DataType dype,
             const paddle::Place& place);

template <typename T>
Tensor empty_like(const Tensor& x,
                  paddle::experimental::DataType dtype,
                  const paddle::Place& place);

// copy tensor for output ptr, in static need use assigh op
template <typename T>
void by_pass(const Tensor& x, Tensor* out);

// set output ptr impl with tmp ptr impl,in dygraph OutGradMeta should be set
template <typename T>
void set_output(const Tensor& x_tmp, Tensor* x);

// These method don't need to be specified
static phi::DDim get_reduce_dims_from_out(const phi::DDim& dout_dims,
                                          const phi::DDim& in_dims) {
  std::vector<int64_t> result;
  int bat = dout_dims.size() - in_dims.size();
  for (int i = 0; i < bat; ++i) {
    result.push_back(i);
  }
  for (int i = 0; i < in_dims.size(); ++i) {
    if (in_dims[i] == 1) {
      result.push_back(i + bat);
    } else {
      PADDLE_ENFORCE_EQ(
          in_dims[i],
          dout_dims[i + bat],
          platform::errors::InvalidArgument(
              "ReduceDims dimension mismatch. Operands could "
              "not be broadcast together with the shape of dout = [%s] and "
              "the shape of in_dims = [%s]. Received [%d] in X is not equal to "
              "[%d] in Y at i:%d.",
              dout_dims,
              in_dims,
              dout_dims[i + bat],
              in_dims[i],
              i));
    }
  }
  return phi::make_ddim(result);
}

static phi::DDim get_reduce_dims(const phi::DDim& x_dims,
                                 const phi::DDim& y_dims) {
  auto out_dims = paddle::operators::details::BroadcastTwoDims(x_dims, y_dims);
  return get_reduce_dims_from_out(out_dims, x_dims);
}

template <typename T>
std::tuple<Tensor, Tensor> modify_dim_for_matmul(const Tensor& a,
                                                 bool is_fold_init_dims_a,
                                                 const Tensor& b,
                                                 bool is_fold_init_dims_b,
                                                 const Tensor* out) {
  Tensor a_out = a;
  Tensor b_out = b;
  bool need_combine =
      (a.dims().size() == 3 || b.dims().size() == 3) && out->dims().size() == 2;
  if (need_combine) {
    auto a_dims = a.dims();
    auto b_dims = b.dims();
    if (is_fold_init_dims_a) {
      if (a_dims.size() == 3) {
        std::vector<int64_t> a_shape = {a_dims[0] * a_dims[1], a_dims[2]};
        a_out = reshape<T>(a_out, IntArray(a_shape));
      }
    } else {
      if (a_dims.size() == 3) {
        a_out = transpose<T>(a, std::vector<int>({1, 0, 2}));
        std::vector<int64_t> a_shape = {a_dims[0], a_dims[1] * a_dims[2]};
        a_out = reshape<T>(a_out, IntArray(a_shape));
      }
    }

    if (is_fold_init_dims_b) {
      if (b_dims.size() == 3) {
        std::vector<int64_t> b_shape = {b_dims[0] * b_dims[1], b_dims[2]};
        b_out = reshape<T>(b_out, IntArray(b_shape));
      }
    } else {
      if (b_dims.size() == 3) {
        b_out = transpose<T>(b, std::vector<int>({1, 0, 2}));
        std::vector<int64_t> b_shape = {b_dims[0], b_dims[1] * b_dims[2]};
        b_out = reshape<T>(b_out, IntArray(b_shape));
      }
    }
  }
  std::tuple<Tensor, Tensor> output(a_out, b_out);
  return output;
}

template <typename T>
void reshape_tensor_to_matrixsequence(
    Tensor* x, const phi::funcs::MatDescriptor& descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }
  if (descriptor.batch_size_) {
    *x = reshape<T>(*x, std::vector<int64_t>({descriptor.batch_size_, h, w}));
  } else {
    *x = reshape<T>(*x, std::vector<int64_t>({h, w}));
  }
}

template <typename T>
void reshape_xyout_to_matrixsequence(
    Tensor* x, Tensor* y, Tensor* out, bool trans_x, bool trans_y) {
  if (x->dims().size() == 1) {
    *x = reshape<T>(*x, std::vector<int64_t>({1, x->dims()[0]}));
  }
  if (y->dims().size() == 1) {
    *y = reshape<T>(*y, std::vector<int64_t>({y->dims()[0], 1}));
  }
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x->dims(), 0, trans_x);
  auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y->dims(), 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    *out = reshape<T>(
        *out, std::vector<int64_t>({mat_dim_x.height_, mat_dim_y.width_}));
  } else {
    *out = reshape<T>(*out,
                      std::vector<int64_t>({(std::max)(mat_dim_x.batch_size_,
                                                       mat_dim_y.batch_size_),
                                            mat_dim_x.height_,
                                            mat_dim_y.width_}));
  }

  reshape_tensor_to_matrixsequence<T>(x, mat_dim_x);
  reshape_tensor_to_matrixsequence<T>(y, mat_dim_y);
}

}  // namespace prim
}  // namespace paddle
