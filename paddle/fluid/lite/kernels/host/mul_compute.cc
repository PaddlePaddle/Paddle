// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <Eigen/Core>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void mul_compute_eigen(const T* x, int x_h, int x_w, const T* y, int y_h,
                       int y_w, T* out) {
  using matrix_t =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Eigen::Map<const matrix_t> X(x, x_h, x_w);
  Eigen::Map<const matrix_t> Y(y, y_h, y_w);
  Eigen::Map<matrix_t> Out(out, x_h, y_w);

  Out = X * Y;
}

class MulCompute : public OpKernel<TARGET(kHost), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulParam;

  void Run() override {
    auto& param = Param<operators::MulParam>();
    core::dim2 x_shape({product(param.x->dims().begin(),
                                param.x->dims().begin() + param.x_num_col_dims),
                        product(param.x->dims().begin() + param.x_num_col_dims,
                                param.x->dims().end())});

    core::dim2 y_shape({product(param.y->dims().begin(),
                                param.y->dims().begin() + param.x_num_col_dims),
                        product(param.y->dims().begin() + param.x_num_col_dims,
                                param.y->dims().end())});

    mul_compute_eigen(param.x->data<float>(), x_shape.x, x_shape.y,  //
                      param.y->data<float>(), y_shape.x, y_shape.y,  //
                      param.output->mutable_data<float>());
    LOG(INFO) << "MUL x " << *param.x;
    LOG(INFO) << "MUL W " << *param.y;
    LOG(INFO) << "MUL out " << *param.output;
  }

  virtual ~MulCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(mul, kHost, kFloat, kNCHW,
                     paddle::lite::kernels::host::MulCompute, def)
    .BindInput("X", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                        TARGET(kHost))})
    .BindInput("Y", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                        TARGET(kHost))})
    .BindOutput("Out", {paddle::lite::Type::Get<paddle::lite::TensorFp32NCHWTy>(
                           TARGET(kHost))})
    .Finalize();
