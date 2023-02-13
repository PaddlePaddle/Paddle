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
#include "paddle/fluid/prim/api/generated_prim/prim_generated_api.h"
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

static std::vector<int> get_reduce_dims(const Tensor& dx,
                                        const int& dout_ndim,
                                        const int& x_ndim,
                                        std::vector<int64_t>* x_dims) {
  // this branch for broadcast with 1dim, we make 1dim to 2dim which make
  // ddout_ndim > dout_dim, but ddout_ndim just can be used when grad_out_grad
  // != nullptr
  if (dout_ndim < x_ndim) {
    return std::vector<int>({});
  }
  const std::vector<std::int64_t> dx_dims = phi::vectorize(dx.dims());
  VLOG(1) << "------------------1---------------";
  std::vector<std::int64_t> broadcast_dims(dout_ndim);
  VLOG(1) << "------------------2---------------";
  std::fill(
      broadcast_dims.data(), broadcast_dims.data() + dout_ndim - x_ndim, 1);
  VLOG(1) << "------------------3---------------";
  std::copy(x_dims->data(),
            x_dims->data() + x_ndim,
            broadcast_dims.data() + dout_ndim - x_ndim);
  VLOG(1) << "------------------4---------------";
  std::vector<int> reduce_dims;
  VLOG(1) << "------------------5---------------";
  for (int i = 0; i <= dout_ndim - 3; i++) {
    if (dx_dims[i] != 1 && broadcast_dims[i] == 1) {
      reduce_dims.push_back(i);
    }
  }
  VLOG(1) << "---------------------------------";
  return reduce_dims;
}

}  // namespace prim
}  // namespace paddle
