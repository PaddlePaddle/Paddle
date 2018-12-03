/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <mkldnn.h>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename T>
class MatMulMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x = context.Input<framework::Tensor>("X");
    auto y = context.Input<framework::Tensor>("Y");

    auto out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto x_dims = x->dims();
    auto y_dims = y->dims();

    auto x_transposed = context.Attr<bool>("transpose_X");
    auto y_transposed = context.Attr<bool>("transpose_Y");

    auto dims = [](const framework::DDim& tensor_dims,
                   const framework::DDim& vector_dims) -> framework::DDim {
      return tensor_dims.size() > 1 ? tensor_dims : vector_dims;
    };

    auto mat_dim_x = math::CreateMatrixDescriptor(dims(x_dims, {1, x_dims[0]}),
                                                  0, x_transposed);
    auto mat_dim_y = math::CreateMatrixDescriptor(dims(y_dims, {y_dims[0], 1}),
                                                  0, y_transposed);

    auto scale = static_cast<T>(context.Attr<float>("alpha"));
    auto beta = T{0};

    auto x_data = x->data<T>();
    auto y_data = y->data<T>();
    auto out_data = out->data<T>();

    auto m = static_cast<int>(mat_dim_x.height_);
    auto n = static_cast<int>(mat_dim_y.width_);
    auto k = static_cast<int>(mat_dim_x.width_);

    auto ldx = x_transposed ? m : k;
    auto ldy = y_transposed ? k : n;

    auto trans_char = [](bool t) -> char { return t ? 'T' : 'N'; };

    auto x_trans_char = trans_char(x_transposed);
    auto y_trans_char = trans_char(y_transposed);

    mkldnn_sgemm(&y_trans_char, &x_trans_char, &n, &m, &k, &scale, y_data, &ldy,
                 x_data, &ldx, &beta, out_data, &n);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(matmul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::MatMulMKLDNNOpKernel<float>);
