/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "MulOp.h"

namespace paddle {

/**
 * mul operator
 * out = scaleT * out + scaleAB*(in1 * in2)
 *
 * \param outputs[0]      output matrix, N * M
 * \param inputs[0]       first input (sparse) matrix,  N * K
 * \param inputs[1]       second input matrix, K * M (non-transpose)
 */
template <DeviceType Device>
class MulFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    scaleAB_ = config.get<real>("scaleAB");
    scaleT_ = config.get<real>("scaleT");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    /// todo(tianbing), add more checks
    CHECK_EQ((size_t)1, inputs.size());
    CHECK_EQ((size_t)2, outputs.size());
    CHECK(inputs[0].data() && inputs[1].data() && outputs[0].data());
    CHECK_EQ(inputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[1].shape().ndims(), (size_t)2);
    CHECK_EQ(outputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);

    CHECK(inputs[0].isSparse()) << "SparseMatrix requried here";
    const auto in1_mat = inputs[0].sparse().SparseMatrix<Device>();
    auto out_mat = outputs[0].matrix<Device>();
    const auto in2_mat = inputs[1].matrix<Device>();
    MulOp<Device>(out_mat, in1_mat, in2_mat, scaleAB_, scaleT_);
  }

private:
  real scaleAB_;
  real scaleT_;
};

#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(MulOp, GPU, MulFunc);
#endif
}  // namespace paddle
