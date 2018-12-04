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

namespace {
template <typename T>
void MKLDNNMatMul(const framework::Tensor& a, const math::MatDescriptor& dim_a,
                  const framework::Tensor& b, const math::MatDescriptor& dim_b,
                  T alpha, framework::Tensor* out, T beta) {
  auto a_data = a.data<T>();
  auto b_data = b.data<T>();
  auto out_data = out->data<T>();

  auto m = static_cast<int>(dim_a.height_);
  auto n = static_cast<int>(dim_b.width_);
  auto k = static_cast<int>(dim_a.width_);

  auto lda = dim_a.trans_ ? m : k;
  auto ldb = dim_b.trans_ ? k : n;

  auto trans_char = [](bool t) -> char { return t ? 'T' : 'N'; };

  auto a_trans_char = trans_char(dim_a.trans_);
  auto b_trans_char = trans_char(dim_b.trans_);

  // MKLDNN sgemm operation uses column-major layout so parameters need to be
  // swapped.
  auto status = mkldnn_sgemm(&b_trans_char, &a_trans_char, &n, &m, &k, &alpha,
                             b_data, &ldb, a_data, &lda, &beta, out_data, &n);
  PADDLE_ENFORCE_EQ(status, mkldnn_success,
                    "MKLDNN sgemm operation executed with no success.");
}
}  // namespace

template <typename T>
class MatMulMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(context.GetPlace()),
                   "It must use CPUPlace.");

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

    auto alpha = static_cast<T>(context.Attr<float>("alpha"));
    auto beta = T{0};

    if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
      MKLDNNMatMul(*x, mat_dim_x, *y, mat_dim_y, alpha, out, beta);
    } else {
      auto& dev_ctx =
          context.template device_context<paddle::platform::CPUDeviceContext>();
      auto blas = math::GetBlas<paddle::platform::CPUDeviceContext, T>(dev_ctx);
      blas.MatMul(*x, mat_dim_x, *y, mat_dim_y, alpha, out, beta);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(matmul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::MatMulMKLDNNOpKernel<float>);
