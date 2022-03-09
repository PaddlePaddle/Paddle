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

#pragma once
#include "glog/logging.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/solve_op.h"
#include "paddle/fluid/operators/tril_triu_op.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
static void triangular_solve(const DeviceContext &context, const Tensor &x,
                             const Tensor &y, Tensor *out, bool upper,
                             bool transpose, bool unitriangular) {
  // Tensor broadcast use eigen library
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) = get_broadcast_dims(x, y);

  Tensor x_bst(x.type());
  TensorExpand<T, DeviceContext>(context, x, &x_bst, x_bst_dims_vec);

  Tensor y_bst(y.type());
  TensorExpand<T, DeviceContext>(context, y, &y_bst, y_bst_dims_vec);

  // TriangularSolveFunctor performs calculations in-place
  // x_clone should be a copy of 'x' after broadcast
  // out should be a copy of 'y' after broadcast
  Tensor x_clone(x.type());
  x_clone.Resize(phi::make_ddim(x_bst_dims_vec));
  x_clone.mutable_data<T>(context.GetPlace());
  framework::TensorCopy(x_bst, context.GetPlace(), context, &x_clone);

  out->Resize(phi::make_ddim(y_bst_dims_vec));
  out->mutable_data<T>(context.GetPlace());
  framework::TensorCopy(y_bst, context.GetPlace(), context, out);

  math::TriangularSolveFunctor<DeviceContext, T> functor;
  functor(context, &x_clone, out, /*left=*/true, upper, transpose,
          unitriangular);
}

template <typename DeviceContext, typename T>
class MatrixReduceSumFunctor {
 public:
  void operator()(const Tensor &input, Tensor *output,
                  const framework::ExecutionContext &ctx);
};

template <typename T>
class MatrixReduceSumFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const Tensor &in, Tensor *out,
                  const framework::ExecutionContext &ctx) {
    // For example: in's dim = [5, 3, 2, 7, 3] ; out's dim = [3, 1, 7, 3]
    // out_reduce_dim should be [0, 2]
    const std::vector<std::int64_t> in_dims = phi::vectorize(in.dims());
    auto in_size = in_dims.size();
    const std::vector<std::int64_t> out_dims = phi::vectorize(out->dims());
    auto out_size = out_dims.size();

    std::vector<std::int64_t> out_bst_dims(in_size);

    std::fill(out_bst_dims.data(), out_bst_dims.data() + in_size - out_size, 1);
    std::copy(out_dims.data(), out_dims.data() + out_size,
              out_bst_dims.data() + in_size - out_size);
    out->Resize(phi::make_ddim(out_bst_dims));

    std::vector<int> out_reduce_dims;
    for (size_t idx = 0; idx <= in_size - 3; idx++) {
      if (in_dims[idx] != 1 && out_bst_dims[idx] == 1) {
        out_reduce_dims.push_back(idx);
      }
    }

    ReduceKernelFunctor<platform::CPUDeviceContext, T, SumFunctor>(
        &in, out, out_reduce_dims, true, false, ctx)
        .template apply<T>();
    out->Resize(phi::make_ddim(out_dims));
  }
};

}  // namespace operators
}  // namespace paddle
