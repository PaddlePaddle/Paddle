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

#include "paddle/fluid/lite/kernels/host/fc_compute.h"
#include <Eigen/Core>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

// NOTE should use pure std C++ implementation.
void FcCompute::Run() {
  using matrix_t = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
  using matrix_map_t = Eigen::Map<matrix_t>;

  auto& param = this->param<param_t>();

  CHECK_EQ(param.in_mat_dims.size(), 2UL);
  CHECK_EQ(param.output->dims().size(), 2UL);
  Eigen::Map<const matrix_t> input(param.input->data<float>(),
                                   param.in_mat_dims[0], param.in_mat_dims[1]);
  Eigen::Map<const matrix_t> weight(param.w->data<float>(), param.w->dims()[0],
                                    param.w->dims()[1]);
  matrix_map_t output(param.output->mutable_data<float>(),
                      param.output->dims()[0], param.output->dims()[1]);

  output = weight.transpose() * input;

  if (param.bias) {
    Eigen::Map<const matrix_t> bias(param.bias->data<float>(),
                                    param.bias->dims()[0],
                                    param.bias->dims()[1]);
    output += bias;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fc, kHost, kFloat, paddle::lite::kernels::host::FcCompute);
